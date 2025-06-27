import streamlit as st
import json
import re
import datetime
from google.cloud import vision
from google.cloud import storage

# Load Google Cloud credentials from Streamlit secrets
try:
    google_cloud_credentials_str = st.secrets["google_cloud_credentials"]
    google_cloud_credentials = json.loads(google_cloud_credentials_str)
except KeyError:
    st.error("Google Cloud credentials not found in Streamlit secrets. Please configure them in .streamlit/secrets.toml")
    st.stop() # Stop the app if credentials are not found

# Initialize Google Cloud Storage client with credentials
storage_client = storage.Client.from_service_account_info(google_cloud_credentials)

# Set up your Google Cloud Storage bucket and project ID
bucket_name = 'ocr-rkmm-docs'
input_folder_name = 'input'
output_folder_name = 'output'
gcs_source_uri = 'gs://ocr-rkmm-docs/input/'
gcs_destination_uri = 'gs://ocr-rkmm-docs/output/'

def main():
    st.title("PDF to TEXT using Google OCR API")

    if "upload_completed" not in st.session_state:
        st.session_state.upload_completed = False
    if "ocr_completed" not in st.session_state:
        st.session_state.ocr_completed = False
    if "display_completed" not in st.session_state:
        st.session_state.display_completed = False
    
    # Upload
    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file is not None:
        if st.button("Upload") and not st.session_state.upload_completed:
            # Ensure the blob path is correct for the uploaded file
            blob = storage_client.bucket(bucket_name).blob(f'{input_folder_name}/{uploaded_file.name}')
            blob.upload_from_file(uploaded_file)
            st.session_state.upload_completed = True
            st.success("File uploaded successfully!")

    if st.session_state.upload_completed:
        if st.button("OCR") and not st.session_state.ocr_completed:
            async_detect_document(f'gs://{bucket_name}/{input_folder_name}/{uploaded_file.name}', f'gs://{bucket_name}/{output_folder_name}/')
            write_to_text(f'gs://{bucket_name}/{output_folder_name}/')
            delete_temporary_files(f'gs://{bucket_name}/{output_folder_name}/')
            st.session_state.ocr_completed = True
            st.success("OCR completed successfully!")

    if st.session_state.ocr_completed:
        if st.button("Display text") and not st.session_state.display_completed:
            with open("transcription.txt", "r", encoding="utf8") as file:        
                content = file.read()
                st.text_area("Text Content", content, height=500)
                st.download_button(label="Download", data=content, file_name="transcription.txt")
            st.session_state.upload_completed = False
            st.session_state.ocr_completed = False
            st.session_state.display_completed = False

def async_detect_document(gcs_source_uri, gcs_destination_uri):
    # Supported mime_types are: 'application/pdf' and 'image/tiff'
    mime_type = 'application/pdf'

    # How many pages should be grouped into each json output file.
    batch_size = 100

    # Initialize Vision AI client with credentials
    vision_client = vision.ImageAnnotatorClient.from_service_account_info(google_cloud_credentials)

    feature = vision.Feature(
        type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)

    gcs_source = vision.GcsSource(uri=gcs_source_uri)
    input_config = vision.InputConfig(
        gcs_source=gcs_source, mime_type=mime_type)

    gcs_destination = vision.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.OutputConfig(
        gcs_destination=gcs_destination, batch_size=batch_size)

    async_request = vision.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config,
        output_config=output_config)

    operation = vision_client.async_batch_annotate_files( # Use vision_client here
        requests=[async_request])

    print('Waiting for the operation to finish.')
    operation.result(timeout=420)


def write_to_text(gcs_destination_uri):

    match = re.match(r'gs://([^/]+)/(.+)', gcs_destination_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)

    bucket = storage_client.get_bucket(bucket_name)

    # List objects with the given prefix.
    blob_list = list(bucket.list_blobs(prefix=prefix))
    print('Output files:')

    transcription = open("transcription.txt", "w")

    for blob in blob_list:
        print(blob.name)

    # Process the first output file from GCS.
    # Since we specified batch_size=2, the first response contains
    # the first two pages of the input file.
    for n in  range(len(blob_list)):
        output = blob_list[n]

        json_string = output.download_as_string()
        response = json.loads(json_string)


        # The actual response for the first page of the input file.
        for m in range(len(response['responses'])):

            first_page_response = response['responses'][m]

            try:
                annotation = first_page_response['fullTextAnnotation']
            except(KeyError):
                print("No annotation for this page.")
                print(KeyError)

            # Here we print the full text from the first page.
            # The response contains more information:
            # annotation/pages/blocks/paragraphs/words/symbols
            # including confidence scores and bounding boxes
            print('Full text:\n')
            print(annotation['text'])
            
            with open("transcription.txt", "a+", encoding="utf-8") as f:
                f.write(annotation['text'])

def delete_temporary_files(gcs_destination_uri):
    """Deletes the output folder and its contents recursively from the bucket."""

    match = re.match(r'gs://([^/]+)/(.+)', gcs_destination_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)

    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    for blob in blobs:
        blob.delete()

    print(f"Folder and its contents deleted from {bucket_name}.")

if __name__ == "__main__":
    main()
