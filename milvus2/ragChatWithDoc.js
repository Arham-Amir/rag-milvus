import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import readline from 'readline';
import * as dotenv from "dotenv";
dotenv.config();


// ########################################
// #### LOGIC TO POPULATE VECTOR STORE ####
// ########################################


const createVectorStore = async () => {
    // Use pdfLoader to scrape content from pdf and create documents
    const loader = new PDFLoader(
        "./ADC_Manufacturing_Pharma_brochure-TEST.pdf"
    );
    const docs = await loader.load();

    // Text Splitter
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 4000,
        chunkOverlap: 20,
    });
    const splitDocs = await splitter.splitDocuments(docs);

    // Instantiate Embeddings function
    const embeddings = new OpenAIEmbeddings({
        model: "text-embedding-3-large",
        dimensions: 1024,
    });

    // Create Vector Store
    const vectorstore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        embeddings
    );

    return vectorstore
}

// ###########################################
// #### LOGIC TO ANSWER FROM VECTOR STORE ####
// ###########################################

const createChain = async (vectorstore) => {
    // Instantiate Model 
    const model = new ChatOpenAI({
        modelName: "gpt-4o",
        temperature: 0.7,
    });

    // Define the prompt for the final chain
    const prompt = ChatPromptTemplate.fromMessages([
        [
            "system",
            "Answer the user's questions based on the following context: {context}.",
        ],
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
    ]);

    // Since we need to pass the docs from the retriever, we will usethe createStuffDocumentsChain
    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt: prompt,
    });
    // Create a retriever from vector store
    const retriever = vectorstore.asRetriever({ k: 2 });

    // Create a HistoryAwareRetriever which will be responsible for generating a search query based on both the user input and the chat history
    const retrieverPrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
        [
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
        ],
    ]);

    // This will merge the list of history messages into single message
    const historyAwareRetriever = await createHistoryAwareRetriever({
        llm: model,
        retriever,
        rephrasePrompt: retrieverPrompt,
    });

    // Create the conversation chain, which will combine the retrieverChain and combineStuffChain in order to get an answer
    const conversationChain = await createRetrievalChain({
        combineDocsChain: chain,
        retriever: historyAwareRetriever,
    });

    return conversationChain
}


// Input from console 
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const askQuestion = (query) => new Promise((resolve) => rl.question(query, resolve));


(async function askFromPdf() {
    let chatHistory = []
    const vectorstore = await createVectorStore()
    const chain = await createChain(vectorstore)
    console.log("\nAsk me anything from pdf or type 'exit' to quit.\n")
    let input;
    input = await askQuestion("User: ");
    while (input.toLowerCase() !== "exit") {
        const response = await chain.invoke({
            input,
            chat_history: chatHistory
        });

        chatHistory.push(new HumanMessage(input));
        chatHistory.push(new AIMessage(response.answer));

        console.log("AI Response:", response.answer);
        // console.log(chatHistory)
        input = await askQuestion("\nUser: ");
    }
    console.log("\n\nThankyou for your time. Good Bye")
})()

