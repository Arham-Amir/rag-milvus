import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MilvusClientManager } from "./milvusUtils.js";

import * as dotenv from "dotenv";
dotenv.config();


// ########################################
// #### LOGIC TO CREATE EMBEDDINGS ####
// ########################################


const getDocumentEmbeddings = async (docName = "./ADC_Manufacturing_Pharma_brochure-TEST.pdf") => {
    // Use pdfLoader to scrape content from pdf and create documents
    const loader = new PDFLoader(
        docName
    );
    const docs = await loader.load();

    // Text Splitter
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 4000,
        chunkOverlap: 20,
    });
    const splitDocs = await splitter.splitDocuments(docs);
    const pagesContentOfDocs = splitDocs.map(doc => doc.pageContent);

    // Instantiate Embeddings function
    const embeddingsModel = new OpenAIEmbeddings({
        model: "text-embedding-3-large",
        dimensions: 1024,
    });

    const embeddings = await embeddingsModel.embedDocuments(pagesContentOfDocs);

    return { embeddings, pagesContentOfDocs }
}

// ###########################################
// #### LOGIC TO Store EMBEDDINGS INTO MILVUS####
// ###########################################

const storeEmbeddingsIntoMilvus = async (embeddings, pagesContentOfDocs) => {
    try {
        const currentTimeDouble = Date.now();
        const milvusClient = new MilvusClientManager()
        const hasCollection = await milvusClient.createOrLoadCollection()
        if (hasCollection) {
            for (let i = 0; i < embeddings.length; i++) {
                // console.log("\n===========================================\n")
                await milvusClient.insertEmbeddingIntoStore([
                    { 'vector': embeddings[i], 'text': pagesContentOfDocs[i], 'timestamp': currentTimeDouble }
                ])
            }
            console.log('Done1')
        }
    } catch (error) {
        console.log("Error storing embeddings: ", error)
    }
}


const { embeddings, pagesContentOfDocs } = await getDocumentEmbeddings("./ADC_Manufacturing_Pharma_brochure-TEST.pdf")
storeEmbeddingsIntoMilvus(embeddings, pagesContentOfDocs)

