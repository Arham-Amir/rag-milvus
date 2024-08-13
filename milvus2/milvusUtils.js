import { MilvusClient, DataType, IndexType, MetricType } from "@zilliz/milvus2-sdk-node";

const MILVUS_ADDRESS = "localhost:19530";
const COLLECTION_NAME = "document_vectors";

export class MilvusClientManager {
    constructor() {
        if (!MilvusClientManager.instance) {
            this.client = new MilvusClient(MILVUS_ADDRESS);
            MilvusClientManager.instance = this;
        }
        return MilvusClientManager.instance;
    }

    async removeCollection() {
        try {
            await this.client.dropCollection({
                collection_name: COLLECTION_NAME,
            });
            // console.log("Collection removed successfully");
        } catch (error) {
            // console.log("Error while removing collection: ", error);
        }
    }

    async describeIndex() {
        const res2 = await this.client.describeIndex({
            collection_name: COLLECTION_NAME,
        })

        // console.log(res2)
    }

    async describeCollection() {
        let res = await this.client.getCollectionStatistics({
            collection_name: COLLECTION_NAME
        })
        console.log(res)
    }

    async _createCollection() {
        const schema = [
            {
                name: "id",
                description: "id field",
                data_type: DataType.Int64,
                is_primary_key: true,
                autoID: true
            },
            {
                name: "vector",
                description: "vector field",
                data_type: DataType.FloatVector,
                type_params: {
                    dim: 1024,
                },
            },
            {
                name: "text",
                description: "document chunks data",
                data_type: DataType.VarChar,
                type_params: {
                    max_length: 65535,
                },
            },
            {
                name: "timestamp",
                description: "embeddings creation time",
                data_type: DataType.Double,
            },
        ];
        const indexParams = {
            collection_name: COLLECTION_NAME,
            field_name: "vector",
            index_type: IndexType.IVF_FLAT,
            metric_type: MetricType.COSINE,
            params: { "nlist": 128 }
        };

        try {
            const res = await this.client.createCollection({
                collection_name: COLLECTION_NAME,
                fields: schema,
                index_params: indexParams,
            });
            // console.log("Collection created ", res.error_code);
            return true
        } catch (error) {
            console.error("Error creating collection:", error);
            return false
        }
    }

    async _loadCollection() {
        try {
            let res = await this.client.loadCollection({
                collection_name: COLLECTION_NAME,
            });
            // Check the load state until it's fully loaded
            let loadState;
            do {
                loadState = await this.client.getLoadState({
                    collection_name: COLLECTION_NAME,
                });
                // console.log("Loading state: ", loadState.state);

                if (loadState.state !== 'LoadStateLoaded') {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            } while (loadState.state !== 'LoadStateLoaded');

            // console.log("Collection loaded successfully.");
            return true
        } catch (error) {
            console.error("Error loading collection:", error);
            return false
        }
    }

    async createOrLoadCollection() {
        const hasCollection = await this.client.hasCollection({
            collection_name: COLLECTION_NAME,
        });
        if (hasCollection.value) {
            return await this._loadCollection();
        } else {
            return await this._createCollection();
        }
    }

    async insertEmbeddingIntoStore(data) {
        try {
            const res = await this.client.insert({
                collection_name: COLLECTION_NAME,
                fields_data: data,
            });
            // console.log("Row inserted successfully into DB.");
        } catch (error) {
            // console.log("Error while inserting row into store: ", error);
        }
    }

    async searchEmbeddingFromStore(vector) {
        try {
            const res = await this.client.search({
                collection_name: COLLECTION_NAME,
                vectors: [vector],
                limit: 3,
                output_fields: ['text']
            });
            // console.log("Data fetched successfully from DB.");
            return res.results
        } catch (error) {
            // console.log("Error while fetching data from store: ", error);
            return []
        }
    }
}

const milvusClientManager = new MilvusClientManager();
milvusClientManager.describeCollection()
// milvusClientManager.createOrLoadCollection();
// milvusClientManager.removeCollection()
