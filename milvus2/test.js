import { MilvusClient } from "@zilliz/milvus2-sdk-node";
import OpenAI from "openai";
import express from "express";
import * as dotenv from "dotenv";
import { DataType } from "@zilliz/milvus2-sdk-node";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import axios from "axios";

dotenv.config();
const app = express();
const port = process.env.PORT || 3000;
app.use(express.json());
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

async function createEmbeddings(input) {
  const embedding = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: input,
    encoding_format: "float",
  });
  return embedding.data[0].embedding;
}
const address = "http://localhost:19530"

const client = new MilvusClient({address});
async function createCollection() {
  const collection_name = "personas";
  client.hasCollection({collection_name})
    .then((exists) => {
      if (exists) {
        console.log(`Collection '${collection_name}' already exists.`);
      } else {
        const fields = [
          {
            name: "id",
            data_type: DataType.Int64,
            is_primary_key: true,
            auto_id: true
          },
          {
            name: "vector",
            data_type: DataType.FloatVector,
            dim: 1536
          },
          {
            name: "persona",
            data_type: DataType.VarChar,
            is_partition_key: true,
            max_length: 600
          },
          {
            name: "details",
            data_type: DataType.VarChar,
            max_length: 65535
          }
        ];
  
        client.createCollection(collection_name, fields)
          .then(() => {
            console.log(`Collection '${collection_name}' created.`);
          })
          .catch((error) => {
            console.error("Error creating collection:", error);
          });
      }
    })
    .catch((error) => {
      console.error("Error checking collection existence:", error);
    });
}
async function splitDescription(description) {
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 300,
        chunkOverlap: 10,
      });
      const chunks = await textSplitter.splitText(description);
      console.log('Splits Length: ', chunks.length);
      console.log('Splits: ', chunks);
      return chunks;
}
async function insertData(data) {
    try {
      const res = await client.insert({
        collection_name: "personas",
        data: data
      });
      console.log("Insert result:", res);
    } catch (error) {
      console.error("Error inserting data:", error);
    }
  }

async function searchData(question, persona) {
    const questionEmbeddings = await createEmbeddings(question);
    const searchParams = {
        collection_name: "personas",
        filter: `persona == "Jake"`,
        vectors: [questionEmbeddings], // Query vector
        output_fields: ["details"], // Fields to return in results
        limit: 1, // Number of results to return
        search_params: {
            metric_type: "L2", // or "IP" for Inner Product
            params: { nprobe: 20 },  // Higher nprobe for more accurate search
            topk: 2
        },
    };
    const searchResult = await client.search(searchParams);
    console.log(searchResult);
    return searchResult;
}
function createDescription({name, age, location, gender, Income, occupation, interest, behaviour}) {
    return `Meet ${name}, a ${age}-year-old ${gender} from ${location}. ${name} works as a ${occupation} and has an income of ${Income}. In their free time, ${name} enjoys ${interest}. ${behaviour}`;
}
app.post('/persona', async (req, res) => {
    try {
        const persona = req.body.name;
        const description = createDescription(req.body);
        console.log(description);
        const splits = await splitDescription(description);
        const data = [];
        for (const split in splits){
            const details = splits[split];
            const vector = await createEmbeddings(details);
            data.push({ vector, persona,details});
        }
        await insertData(data);
        res.json({ response: `${data.length} Embeddings Inserted` });
    } catch (error) {
        console.log(error);
        res.json({ response: error.message });
    }
});

app.get('/persona/search', async (req, res) => {
    try {
        const {question, persona} = req.body;
        const result = await searchData(question, persona);
        res.json({ response: result });
    } catch (error) {
        console.log(error);
        res.json({ response: error.message });
    }
});
async function loadCollection() {
  try {
    await client.loadCollection({
      collection_name: "personas"
    });
    console.log("Collection loaded successfully");
  } catch (error) {
    console.error('Failed to load collection:', error);
  }
}
async function releaseCollection() {
  try {
    await client.releaseCollection({
      collection_name: "personas"
    });
    console.log("Collection released");
  } catch (error) {
    console.error('Failed to release collection:', error);
  }
}
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
async function queryAllData() {
  try {
    // await client.loadCollection({
    //   collection_name: "personas"
    // });

    const res1 = await client.getLoadState({
      collection_name: "personas"
    });
    console.log("Load state:", res1.state);

    while (res1.state !== "LoadStateLoaded") {
      await sleep(1000);
      const res2 = await client.getLoadState({
        collection_name: "personas"
      });
      console.log("Load state:", res2.state);
    }

    const res = await client.query({
      collection_name: "personas",
      filter: "",
      output_fields: ["*"],
      limit: 10000
    });
    console.log(res);
  } catch (error) {
    console.error("Error querying data:", error);
  }// finally {
  //   await releaseCollection();
  // }
}

async function describeCollection() {
  try {
    const res = await client.describeCollection({
      collection_name: "personas"
    });
    console.log(res);
  } catch (error) {
    console.error('Failed to describe collection:', error);
  }
}

async function checkServerStatus() {
  try {
    // Attempt to list collections
    const response = await client.listCollections();
    console.log("Milvus server is running and responsive.");
    console.log("Collections:", response.collection_names);
} catch (error) {
    console.error("Error connecting to Milvus server:", error.message);
} 
}

await checkServerStatus();
// await releaseCollection();
// await describeCollection();
await createCollection();
await queryAllData();