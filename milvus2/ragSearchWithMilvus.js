import { OpenAIEmbeddings } from "@langchain/openai";
import { MilvusClientManager } from "./milvusUtils.js";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import readline from 'readline';
import * as dotenv from "dotenv";
dotenv.config();

// ########################################
// #### LOGIC TO FETCH EMBEDDINGS ####
// ########################################

const createQueryEmbeddings = async (query) => {
  const embeddingsModel = new OpenAIEmbeddings({
    model: "text-embedding-3-large",
    dimensions: 1024,
  });

  const embeddings = await embeddingsModel.embedDocuments([query]);
  return embeddings[0];
};

const searchEmbeddingInMilvus = async (milvusClient, embedding) => {
  try {
    return await milvusClient.searchEmbeddingFromStore(embedding);
  } catch (error) {
    console.log("Error searching embeddings: ", error);
    return [];
  }
};

// ###########################################
// #### LOGIC TO ANSWER FROM MILVUS DATA ####
// ###########################################

const createAndCallChain = async (chatHistory, query, mergedText) => {
  // Instantiate Model 
  const model = new ChatOpenAI({
    modelName: "gpt-4",
    temperature: 0.7,
  });

  // Define the prompt for the final chain
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's questions based on the following context: {context}.",
    ],
    new MessagesPlaceholder("chatHistory"),
    ["user", "{input}"],
  ]);

  const chain = prompt.pipe(model)

  const response = await chain.invoke({
    input: query,
    context: mergedText,
    chatHistory: chatHistory,
  });
  chatHistory.push(new HumanMessage(query));
  chatHistory.push(new AIMessage(response.content));
  return response.content;
};

// Input from console 
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
const askQuestion = (query) => {
  return new Promise((resolve) => rl.question(query, resolve));
}

(async function askFromPdf() {
  const milvusClient = new MilvusClientManager();
  const hasCollection = await milvusClient.createOrLoadCollection();
  if (hasCollection) {
    let chatHistory = [];
    console.log("\nAsk me anything from pdf or type 'exit' to quit.\n");
    let input;
    input = await askQuestion("User: ");
    while (input.toLowerCase() !== "exit") {
      let embedding = await createQueryEmbeddings(input);
      let retrievedDocs = await searchEmbeddingInMilvus(milvusClient, embedding);
      let mergedText = retrievedDocs.map(doc => doc.text).join(' ');
      const response = await createAndCallChain(chatHistory, input, mergedText);

      console.log("AI Response:", response);
      input = await askQuestion("\nUser: ");
    }
    console.log("\n\nThank you for your time. Goodbye");
  }
  else{
    console.log("Collection not exist.")
  }
})();
