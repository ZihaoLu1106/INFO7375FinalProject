# 📖 Storytelling Model Fine-Tuning

This project fine-tunes a pre-trained **tiny GPT-2** model to generate full stories from short prompts, using the **Writing Prompts** dataset from Kaggle.  
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
