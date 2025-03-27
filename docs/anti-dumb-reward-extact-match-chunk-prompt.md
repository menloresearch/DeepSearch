# Anti-dumb reward extact match chunk prompt

@reward-functions.md  @train_autodidact_1B.py  @rl_helpers.py  

I need to implement this function, you check the idea in @reward-functions.md . the function need to somehow be able to compare the grouth truth document chunk that the question and answer is created from, which is

- data in data/data_v1/saved_data/questions.json
- data sample:

  ```
    {
    "chunk_id": 1,
    "question": "What was the location of the first pad abort of the mission?",
    "answer": "White Sands Missile Range",
    "difficulty": "easy"
  },
  ```

- chunk content in data/data_v1/saved_data/chunks.pkl
- chunk id is mapped to the chunk content
- im dumb please make it easy for me to implement
