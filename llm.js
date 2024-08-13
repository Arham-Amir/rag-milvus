import { ChatOpenAI } from "@langchain/openai";

import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";

import dotenv from 'dotenv';
dotenv.config();

// Instantiate Model
const model = new ChatOpenAI({
    model: "gpt-4o",
});

// Create prompt
const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the user's question from the following context: 
  {context}
  Question: {input}`
);

// Create Chain
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
});

// fetch data from pdf
const loader = new PDFLoader("./ADC_Manufacturing_Pharma_brochure-TEST.pdf");
const docs = await loader.load();

// Text Splitter
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 4000,
    chunkOverlap: 20,
});
const splitDocs = await splitter.splitDocuments(docs);

// // Instantiate Embeddings function
const embeddings = new OpenAIEmbeddings();

// // Create Vector Store
const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);

// // Create a retriever from vector store
const retriever = vectorstore.asRetriever({ k: 2 });

// // Create a retrieval chain
const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever,
});


const response = await retrievalChain.invoke({
    input: "What is agenda?",
});

console.log(response);