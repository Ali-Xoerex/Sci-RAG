# Different Types of RAG systems
1. ## Naive RAG
  * They leverage **static knowledge bases**, meaning they retrieve knowledge from a pre-defined, fixed dataset.
  * They are **Single-Step** retrievers. They return the first k relavent chunks in their dataset in one step.
  * They are **Zero-Shot** RAG systems, meaning they don't need additional fine-tuning. We use pre-trained models, out-of-the-box style.
  * They usually are **Local** systems, meaning their knowledge base reside on local memory or disk. 
2. ## Dual RAG
  * They still are **static knowledge based**, **Single-Step**, **Zero-Shot** and **Local**.
  * However, they use **two** encoders. One for queries and one for context.
  * They use Dense Retrieval systems for better efficiency, when the knowledge base is huge.
  * They are **Modular**, since encoder and generator models are separate.
