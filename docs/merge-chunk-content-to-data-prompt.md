I'll give you two file below. your job is to create a script that bring the content of the chunk file to the question file, map by the chunk_id, which is the sequential number of the chunk in the chunk file. the new column should be called "chunk_content".

/home/thinhlpg/code/DeepSearch/data/data_v1/saved_data/questions.json
[
  {
    "chunk_id": 1,
    "question": "What was the location of the first pad abort of the mission?",
    "answer": "White Sands Missile Range",
    "difficulty": "easy"
  },

  /home/thinhlpg/code/DeepSearch/data/data_v1/saved_data/chunks.pkl
