import json
from urllib.parse import unquote
import hashlib
from .text_splitter import split_text_into_chunks
from .excel_splitter import split_excel_into_chunks, generate_hash, split_edms_excels_into_chunks, split_csv_file_into_chunks
from .python_module_splitter import script_splitter
from .document_intelligence_splitter import (
    split_powerpoint_pdf_into_chunks,
    split_ppt_pdf_lakehouse,
    split_word_documents_into_chunks,
    split_edms_pptx_documents,
    split_edms_word_documents,
    split_edms_pdf_documents,
    extract_doc_pages,
    split_ppt_into_chunks
)
from .text2sql_metadata_splitter import (
    metadata_splitter_from_blob,
    metadata_splitter_from_blob_new, 
    read_sample_queries_from_blob, 
    read_ddl_metadata_from_blob, 
    read_sample_queries_from_blob_updated,
    read_new_text2sql_metadata_from_blob,
    read_new_text2sql_metadata_v2_from_blob,
    read_metadata_from_blob_updated,
    read_material_drawing_data_from_blob
)
from .fmea_metadata_splitter import extract_fields_from_blob_fmea_agent, extract_fields_from_blob_fmea_agent_updated,extract_fields_from_blob_fmea_poc, extract_fields_from_blob_fmea_agent_conversational
from .pdf_splitter import split_pdf_file_into_chunks
from .logging import logger
import asyncio
import time
from .constants import FileExtenstions
from .p2f_json_splitter import process_json_file 
# Excel with LLM Generated Summary Module
from .excel_splitter_with_summary import generateExcelChunksWithSummary
from .wbt_metadata_excel_processor import wbt_metadata_splitter_from_blob
from dotenv import load_dotenv
load_dotenv()

async def split_text_single(value, size, overlap, semaphore):
    # This is used in split_text_batch

    # Validate the input
    try:
        assert "recordId" in value
        record_id = value["recordId"]
    except AssertionError:
        return None

    # Validate the input
    try:
        assert "data" in value, "Field 'data' is required."
        data = value["data"]
        for field in ["parent_filename", "blob_uri"]:
            assert field in data, f"Field '{field}' is required."
    except AssertionError as error:
        return {
            "recordId": record_id,
            "data": {},
            "errors": [{"message": f"Error: {error.args[0]}"}],
        }

    # Perform the operations
    parent_filename = data["parent_filename"]
    blob_uri = data["blob_uri"]
    content = data.get("content", "") 

    logger.info("====================File Metadata====================")
    logger.info(f"parent_filename = {parent_filename}")
    logger.info(f"blob_uri = {blob_uri}")
    logger.info("====================File Metadata====================")

    is_processed_using_custom_processing = False
    warning = []
    error_list = []

    print("Processing: ", blob_uri)
    try:
        if (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "customerSurvey" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="customerSurvey",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,FileExtenstions.XLSM.value))
            and ("fmea_meta_sharepoint" in blob_uri)
        ):
            chunks_with_metadata = extract_fields_from_blob_fmea_poc(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,FileExtenstions.XLSM.value))
            and ("wbt-metadata" in blob_uri)
        ):
            chunks_with_metadata = wbt_metadata_splitter_from_blob(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True

        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and (("LamIndiaFinancePOC/Travel" in blob_uri) or ("LamIndiaFinancePOC/Finance" in blob_uri))
        ):
            chunks_with_metadata = await generateExcelChunksWithSummary(
                blob_uri=blob_uri, 
                chunk_size=size,
                chunk_overlap=overlap
            )
            is_processed_using_custom_processing = True

        elif (
            parent_filename.lower().endswith((FileExtenstions.PDF.value))
            and (("LamIndiaFinancePOC/Travel" in blob_uri) or ("LamIndiaFinancePOC/Finance" in blob_uri))
        ):
            chunks_with_metadata = await split_powerpoint_pdf_into_chunks(
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
                semaphore=semaphore,
            )
            is_processed_using_custom_processing = True

        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "commonSpec" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="commonSpec",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "servicedesk" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="servicedesk",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.PDF.value))
            and (("servicedesk" in blob_uri) or ("corporateAccounting" in blob_uri) or ("PayrollKB" in blob_uri))
        ):
            chunks_with_metadata = split_pdf_file_into_chunks(
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "corporateAccounting" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="corporate-accounting",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.CSV.value))
            and ("servicedesk" in blob_uri or "semis2_cci" in blob_uri)
        ):
            chunks_with_metadata = split_csv_file_into_chunks(
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "drAttachments" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="drAttachments",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "DPG_modeling" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="modelling-report",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "semis2_cci" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="semis2_cci",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "sabre3d_kpr" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="sabre3d_kpr",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif(
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "sabre3d_bdsite" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="sabre3d_bdsite",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif(
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "sabre3d_epl" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="sabre3d_epl",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((".json"))
            and "P2F/output" in blob_uri
        ):
            chunks_with_metadata = process_json_file(
                blob_uri=blob_uri,
                usecase="p2f"
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "demodata-cci" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="demo-data",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "iplmfaq" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="iplm-faq",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "DPG_reliability" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="reliability-report",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.PPTX.value))
            and (("onelake" in blob_uri) and ("DPG" in blob_uri))
        ):
            print("Processing DPG onelake pptx")
            chunks_with_metadata = await split_ppt_pdf_lakehouse(
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
                semaphore=semaphore,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "coAttachments" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="coAttachments",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "partAttachments" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="partAttachments",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "prAttachments" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="prAttachments",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLS.value, FileExtenstions.XLSX.value))
            and "DigitalTransformation" in blob_uri
        ):
            chunks_with_metadata = split_excel_into_chunks(
                usecase="DigitalTransformation",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value, FileExtenstions.XLSM.value))
            and (("edmsengineeringstandards" in blob_uri.lower()) or ("edmsnonengineeringstandards" in blob_uri.lower()) or ("edms" in blob_uri.lower()) or ("demoadlsfolder3/llamaindex-samples/iplm" in blob_uri.lower()))
        ):
            chunks_with_metadata, warning_msg = split_excel_into_chunks(
                usecase="edms",
                blob_uri=blob_uri,
                chunk_size=size,
                chunk_overlap=overlap,
            )
            if chunks_with_metadata is not None:
                is_processed_using_custom_processing = True
            else:
                error_list.append(
                    {
                        "message": f"Could not process edms excel file. Reason:{warning_msg}. File Path: {blob_uri}"
                    }
                )
        elif (
            parent_filename.lower().endswith((FileExtenstions.PDF.value))
            and (("edmsengineeringstandards" in blob_uri.lower()) or ("edmsnonengineeringstandards" in blob_uri.lower()) or ("edms" in blob_uri.lower()) or ("confluence/sem_3d_poc" in blob_uri.lower()))
        ):
            chunks_with_metadata = await split_edms_pdf_documents(
                blob_uri=blob_uri, chunk_size=size, include_metadata_in_chunk=True, process_images=True, chunk_overlap=overlap, semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif(
            parent_filename.lower().endswith((FileExtenstions.DOCX.value))
            and (("edmsengineeringstandards" in blob_uri.lower()) or ("edmsnonengineeringstandards" in blob_uri.lower()) or ("edms" in blob_uri.lower()) or ("ce-documents" in blob_uri.lower()) or ("LamIndiaFinancePOC/Travel" in blob_uri) or ("LamIndiaFinancePOC/Finance" in blob_uri) or ("demoadlsfolder3/llamaindex-samples/iplm" in blob_uri.lower()))
        ):
            chunks_with_metadata = await split_edms_word_documents(
                blob_uri=blob_uri, include_metadata_in_chunk=True, process_images=True, chunk_size=size, chunk_overlap=overlap, semaphore=semaphore
            )
            if chunks_with_metadata is not None:
                is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.PPTX.value,FileExtenstions.PPT.value))
            and (("edmsengineeringstandards" in blob_uri.lower()) or ("edmsnonengineeringstandards" in blob_uri.lower()) or ("edms" in blob_uri.lower()) or ("LamIndiaFinancePOC/Travel" in blob_uri) or ("LamIndiaFinancePOC/Finance" in blob_uri) or ("demoadlsfolder3/llamaindex-samples/iplm" in blob_uri.lower()))
        ):
            chunks_with_metadata = await split_edms_pptx_documents(
                blob_uri=blob_uri, include_metadata_in_chunk=True, process_images=True, semaphore=semaphore)
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.PDF.value, FileExtenstions.PPT.value, FileExtenstions.PPTX.value))
            and (("iPLM" in blob_uri) or ("demoadlsfolder3/llamaindex-samples/iplm" in blob_uri.lower()))
        ):
            chunks_with_metadata = await split_powerpoint_pdf_into_chunks(
                blob_uri=blob_uri, chunk_size=size, chunk_overlap=overlap,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.DOC.value, FileExtenstions.DOCX.value))
            and (("LightCAEBot" in blob_uri) or ("drAttachments" in blob_uri) or ("coAttachments" in blob_uri) or ("partAttachments" in blob_uri) or ("prAttachments" in blob_uri))
        ):
            chunks_with_metadata = await split_word_documents_into_chunks(
                blob_uri=blob_uri, chunk_size=size, chunk_overlap=overlap,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.DOC.value, FileExtenstions.DOCX.value))
                and (("DPG_modeling" in blob_uri) or ("DPG_reliability" in blob_uri) or ("demodata-cci" in blob_uri) or ("iplmfaq" in blob_uri) or ("semis2_cci" in blob_uri) or ("sabre3d_epl" in blob_uri) or ("sabre3d_bdsite" in blob_uri) or ("sabre3d_kpr" in blob_uri))
        ):
            chunks_with_metadata = await extract_doc_pages(
                blob_uri=blob_uri,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.PDF.value, FileExtenstions.PPT.value, FileExtenstions.PPTX.value))
            and (("LightCAEBot" in blob_uri) or ("drAttachments" in blob_uri) or ("coAttachments" in blob_uri) or ("partAttachments" in blob_uri) or ("prAttachments" in blob_uri) or ("DPG_modeling" in blob_uri) or ("DPG_reliability" in blob_uri) or ("demodata-cci" in blob_uri) or ("iplmfaq" in blob_uri) or ("vizglow" in blob_uri.lower()) or ("semis2_cci" in blob_uri) or ("sabre3d_epl" in blob_uri) or ("sabre3d_kpr" in blob_uri))
        ):
            chunks_with_metadata = await split_powerpoint_pdf_into_chunks(
                blob_uri=blob_uri, chunk_size=size, chunk_overlap=overlap,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.PDF.value, FileExtenstions.PPTX.value))
        and ("sabre3d_bdsite" in blob_uri)
        ):
            chunks_with_metadata = await split_powerpoint_pdf_into_chunks(
                blob_uri=blob_uri, chunk_size=size, chunk_overlap=overlap,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.PPT.value,))
            and ("sabre3d_bdsite" in blob_uri)
        ):
            chunks_with_metadata = await split_ppt_into_chunks(
                blob_uri=blob_uri, chunk_size=size, chunk_overlap=overlap,semaphore=semaphore
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value))
            and "iPLM/text2sql_metadata" in blob_uri
        ):
            chunks_with_metadata = metadata_splitter_from_blob_new(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,))
            and "iPLM/text2sql_ddl" in blob_uri
        ):
            chunks_with_metadata = read_ddl_metadata_from_blob(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,))
            and "iPLM/text2sql_sample_queries" in blob_uri
        ):
            chunks_with_metadata = read_sample_queries_from_blob(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,))
            and "iPLM/text2sql_new_metadata" in blob_uri
        ):
            chunks_with_metadata = read_new_text2sql_metadata_v2_from_blob(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,))
            and "iPLM/parts_drawing" in blob_uri
        ):
            chunks_with_metadata = read_material_drawing_data_from_blob(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.PY.value,))
            and "gemini" in blob_uri
        ):
            chunks_with_metadata = script_splitter(blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,))
            and ("nce_text2sql/text2sql_metadata" in blob_uri or "ehs_text2sql/text2sql_metadata" in blob_uri)
        ):
            chunks_with_metadata = read_metadata_from_blob_updated(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,))
            and ("nce_text2sql/text2sql_ddl" in blob_uri or "ehs_text2sql/text2sql_ddl" in blob_uri or "hr_text2sql/ddl" in blob_uri)
        ):
            chunks_with_metadata = read_ddl_metadata_from_blob(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,))
            and ('nce_text2sql/text2sql_sample_queries' in blob_uri or 'ehs_text2sql/text2sql_sample_queries' in blob_uri)
        ):
            chunks_with_metadata = read_sample_queries_from_blob_updated(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,))
            and "hr_text2sql/metadata" in blob_uri
        ):
            chunks_with_metadata = metadata_splitter_from_blob(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,))
            and "hr_text2sql/samplequeries" in blob_uri
        ):
            chunks_with_metadata = read_sample_queries_from_blob(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,))
            and "iPLM/fmea-agent" in blob_uri
        ):
            chunks_with_metadata = extract_fields_from_blob_fmea_agent_updated(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        elif (
            parent_filename.lower().endswith((FileExtenstions.XLSX.value,FileExtenstions.XLSM.value))
            and ("fmea_cci" in blob_uri)
        ):
            chunks_with_metadata = extract_fields_from_blob_fmea_agent_conversational(
                blob_uri=blob_uri
            )
            is_processed_using_custom_processing = True
        
    except Exception as e:
        warning = [
            {
                "message": f"Could not complete operation for record using Custom Processing. Reason: {str(e)}. File Path: {blob_uri}"
            }
        ]

    if not is_processed_using_custom_processing:
        chunks = split_text_into_chunks(
            content=content, chunk_size=size, chunk_overlap=overlap
        )
        unquoted_blob_uri = unquote(blob_uri)

        if "blob.core.windows.net" in unquoted_blob_uri:
            url_to_split_on = ".blob.core.windows.net/"
            parent_path_with_container_name = unquoted_blob_uri.split(url_to_split_on)[1]
            use_case = parent_path_with_container_name.split("/")[1]
        elif "dfs.fabric.microsoft.com" in unquoted_blob_uri:
            url_to_split_on = ".dfs.fabric.microsoft.com/"
            parent_path_with_container_name = unquoted_blob_uri.split(
                "https://onelake.dfs.fabric.microsoft.com/"
            )[-1]
            use_case = parent_path_with_container_name.split("/")[3]
        else:
            use_case = "default-fabric"
            parent_path_with_container_name = unquoted_blob_uri.split(
                "https://onelake.dfs.fabric.microsoft.com/"
            )[-1]
        chunks_with_metadata = []
        for text in chunks:
            data = {
                "content": text,
                "chunk_hash": generate_hash(text),
                "use_case": use_case,
                "sheet_name": "",
                "row": -1,
                "parent_path": parent_path_with_container_name,
            }
            chunks_with_metadata.append(data)

    return {
        "recordId": record_id,
        "data": {"chunks": chunks_with_metadata},
        "errors": error_list,
        "warnings": warning,
    }


async def split_text_batch(values, size, overlap):
    """
    Custom split text function
    """
    response = {}
    # setting the concurrency to 5
    semaphore = asyncio.Semaphore(5)
    response["values"] = []
    print("Length: ", len(values))
    start_time = time.time()
    tasks = [asyncio.create_task(split_text_single(value, size, overlap, semaphore)) for value in values] 
    results = await asyncio.gather(*tasks)
    print("Time taken: ", time.time() - start_time)
    response["values"] = results
    return json.dumps(response)
