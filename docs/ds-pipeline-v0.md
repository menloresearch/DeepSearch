# Dataset pipeline v0

- Why not just create whole new dataset?
    - we want to keep the same dataset for training and evaluation
    - because the initial dataset is already good
    - we don't want to waste it

- Goal: introduce paraphrased document chunks to the training process
- Ok let just go with the plan below cuz it's FAST to implement!s
    - Smol model 0.5b
    - Simple prompt 3 prompts -> 3 paraphrased chunks for each original chunk (why 3? idk, it was revealed for me in my dream, but it's smol and fast to run)
        - short medium long
        - 3 different styles/ personalities

- Next (v0.1):
    - Try this  <https://github.com/argilla-io/synthetic-data-generator>

## How?

- Please refer to [250324_generate_data_anatomy.ipynb](../notebooks/250324_generate_data_anatomy.ipynb) for more details
    - There are already 3 files generated by original `generate_dataset.py` script. There are chunk id in the question json file.
    - should modify the `chunks` file to include paraphrased chunks
    - re run faiss index

- Final data has "chunk_id" field in the question json file, is it used or is it important for the training process or evaluation? - no (checked with Ctrl + F), only the "question" and "answer" matter -> **so i can just iterate over the chunk file and add paraphrased chunks to the vector store**
    - How do i iterate over the `chunk.pkl` file?
        - use pickle to load the file
        - iterate over the file
        - paraphrase the chunk [paraphrase-prompt.md](paraphrase-prompt.md)
        - add the paraphrased chunks to the vector store (how? will it affect the original chunk id?)
            - Can just append the new chunks to the existing file - Yes, but:
                - The original vectors (first 10 in your example) KEEP their IDs (0-9)
                - New vectors (last 10) get new IDs (10-19)
        - save the vector store
        - save the question json file
- [ ] Should I ass wrong information or not? How correct should the paraphrased chunk be? How many paraphased chunks should I add for each original chunk? - **V0.1? for now just use simple paraphrasing with correct information.**
