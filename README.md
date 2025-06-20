# 📖 Storytelling Model Fine-Tuning and RAG

This project fine-tunes a pre-trained **tiny GPT-2** model to generate full stories from short prompts, using the **Writing Prompts** dataset from Kaggle. And use style stories and paraphrase-MiniLM-L6-v2 as RAG
The model learns to take creative ideas and expand them into rich, detailed narratives!

---

## 🛠 Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/ZihaoLu1106/INFO7375FinalProject.git
cd INFO7375FinalProject
```

2. **Python Environemt or Create a Virtual Environment**
```bash
python version:3.9.6
```
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Required Libraries**
```bash
pip install torch transformers datasets kagglehub pandas
```

> ✅ Note:  
> - `torch` for model training  
> - `transformers` for loading GPT-2  
> - `datasets` for HuggingFace dataset handling  
> - `kagglehub` for dataset download  
> - `pandas` for data handling

---

## 📚 Dataset

We use the [**Writing Prompts**](https://www.kaggle.com/datasets/ratthachat/writing-prompts) dataset from Kaggle, which contains:
- **Prompts**: Short creative ideas
- **Targets**: Full-length fictional stories

The dataset is downloaded automatically using `kagglehub`.
Data sample:

                                              source  \
 0  [ WP ] You 've finally managed to discover the...   
 1  [ WP ] The moon is actually a giant egg , and ...   
 2  [ WP ] You find a rip in time walking through ...   
 3  [ WP ] For years in your youth the same imagin...   
 4  [ WP ] You glance at your watch 10:34 am , rou...   
 
                                               target  
 0  So many times have I walked on ruins , the rem...  
 1  -Week 18 aboard the Depth Reaver , Circa 2023-...  
 2  I was feckin ' sloshed , mate . First time I e...  
 3  “ No , no no no ... ” She backed up and turned...  
 4  There 's a magical moment between wakefulness ...  
 
                                               source  \
 0  [ WP ] Every person in the world undergoes a `...   
 1  [ WP ] Space mining is on the rise . The Space...   
 2  [ WP ] `` I wo n't have time to explain all of...   
 3  [ CW ] Write about a song . Each sentence must...   
 4  [ EU ] You live in Skyrim . It is your job to ...   
 
                                               target  
 0  Clancy Marguerian , 154 , private first class ...  
 1  „… and the little duckling will never be able ...  
 2  I wo n't have the time to explain all of this ...  
 3  * '' [ Sally ] ( https : //www.youtube.com/wat...  
 4  Light is a marvelous thing . It alone can turn...  

---
## 📚 Model

We use the [**Tiny GPT-2**](https://huggingface.co/sshleifer/tiny-gpt2) from Huggingface

---
## 📚 RAG

We use the [**paraphrase-MiniLM-L6-v2**](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) from Huggingface

---

## 🚀 Project Structure

| File | Description |
|:---|:---|
| `StoryTellingModel.ipynb` | Main notebook for training and evaluation |
| `outputs/` | Folder for saved models and logs (created after training) |

---

## ⚙️ Main Steps

1. **Import Libraries**  
2. **Download Dataset**  
3. **Load and Preprocess Data**  
4. **Initialize Tokenizer**  
5. **Tokenize Dataset**  
6. **Prepare Model and Trainer**  
7. **Train the Model**

---

## 📈 Example Usage (after training)

```python
# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("./results")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate a story from a prompt
prompt = "A young girl discovers a hidden door in her backyard."
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=300, num_return_sequences=1)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```
```python
# output
[ WP ] The world consists of `` users '', `` builders '', and `` thinkers ''. You are a `` thinker '', a physicist on the brink of proving a new abstract theory. On the decisive day you prove this theory you've become the first known `` master ''.
 [ WP ] This is my first post here, sorry if this has been done before. I have a very detailed and expansive prompt. I came up with this a while ago, and it was too ambitious to tackle myself. I feel like it has too much potential to just be forgotten. Enjoy!
 [ CW ] Go to this website : http : //imgurroulette.com/ click `` letsgo '', and write a short story about whatever appears. Include the direct link to the image in your reply.
 A young girl discovers a hidden door in her backyard. '',., you the you to `` you mynew < the thenew.. ``, < the `` the and <,> the in a., the to The I,., I Iline>new.> that the,new you � in the <new '' I a was. ``.>> andnew to of't'you.>line of.,. on the.newline <, his the,.new his..new. ''lineline.> `` me you forlines the>.> in>lineline < the <>, is it a Iline> to.s in the of andnewnewnew '' is>new his I> the <..> <
```

---

## 💬 Future Improvements

- Fine-tune with larger GPT-2 variants
- Apply Top-p (nucleus) sampling during generation
- Deploy the model into a web application for interactive storytelling

---

## 🙌 Acknowledgements

- HuggingFace Transformers for model loading and training
- Kaggle community for dataset resources

---

# ✨ Let's bring ideas to life with AI-driven storytelling!
