import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers.data.processors.utils import InputFeatures
from transformers import Trainer, TrainingArguments, GPT2Tokenizer, AutoModel, AutoTokenizer, TextDataset
from transformers import pipeline, set_seed, DataCollatorForLanguageModeling, AutoModelForCausalLM

DATA_PATH = 'C:\\Users\\Chambers\PycharmProjects\\foo\\venv\\recent_raw_tweets.csv'
OUTPUT_DIRECTORY = 'raw_DJT_recent'
MODEL_NAME = 'raw_DJT_recent'

class TweetDataset(Dataset):
    def __init__(self, tweets, tokenizer, max_len):
        self.tweets, self.tokenizer, self.max_len = tweets.to_numpy(), tokenizer, max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = self.tweets[item]
        tokens = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return InputFeatures(input_ids=tokens['input_ids'].flatten().long().numpy().tolist(),
                             attention_mask=tokens['attention_mask'].flatten().long().numpy().tolist())

def create_dataset(df, tokenizer, max_length, batch_size):
    return TweetDataset(df, tokenizer, max_length)

def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)
    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


# Load GPT2 Pieces
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

PAD = "I DON’T do it for the money. I’ve got enough, much more than I’ll ever need. I do it to do it. Deals are my art form. Other people paint beautifully on canvas or write wonderful poetry. I like making deals, preferably big deals. That’s how I get my kicks. Most people are surprised by the way I work. I play it very loose. I don’t carry a briefcase. I try not to schedule too many meetings. I leave my door open. You can’t be imaginative or entrepreneurial if you’ve got too much structure. I prefer to come to work each day and just see what develops. There is no typical week in my life. I wake up most mornings very early, around six, and spend the first hour or so of each day reading the morning newspapers. I usually arrive at my office by nine, and I get on the phone. There’s rarely a day with fewer than fifty calls, and often it runs to over a hundred. In between, I have at least a dozen meetings. The majority occur on the spur of the moment, and few of them last longer than fifteen minutes. I rarely stop for lunch. I leave my office by six-thirty, but I frequently make calls from home until midnight, and all weekend long. It never stops, and I wouldn’t have it any other way. I try to learn from the past, but I plan for the future by focusing exclusively on the present. That’s where the fun is. And if it can’t be fun, what’s the point?"

# Customize the Tokenizer
QS = '<QS>' # quote start
QE = '<QE>' # quote end
RS = '<RS>' # reply start
RE = '<RE>' # reply end
RT = '<RT>' # retweet
AT = '<AT>' # @ mentions

SPECIAL_TOKENS = {'additional_special_tokens': [QS, QE, RS, RE, RT, AT]}

num_added_toks = tokenizer.add_special_tokens(SPECIAL_TOKENS)
num_added_toks_2 = tokenizer.add_special_tokens({'pad_token': f'[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Load dataset
df = pd.read_csv(DATA_PATH)

# Split data
train_df, val_df = train_test_split(df['text'], test_size=0.1, random_state=42)
train_dataset = create_dataset(train_df, tokenizer, 512, 2048)
eval_dataset = create_dataset(val_df, tokenizer, 512, 2048)

train_df = list(train_df)
val_df = list(val_df)

train_mod_path = "train.csv"
test_mod_path = "test.csv"

train_m = ""
for tweet in train_df:
    train_m += (tokenizer.special_tokens_map['bos_token'] + tweet.rstrip() + tokenizer.special_tokens_map['eos_token'])

with open(train_mod_path, "w", encoding='utf-8') as f:
    f.write(train_m)

test_m = ""
for tweet in val_df:
    test_m += (tokenizer.special_tokens_map['bos_token'] + tweet.rstrip() + tokenizer.special_tokens_map['eos_token'])

with open(test_mod_path, "w", encoding='utf-8') as f:
    f.write(test_m)

train_dataset, test_dataset, data_collator = load_dataset(train_mod_path, test_mod_path, tokenizer)

training_args = TrainingArguments(
    output_dir='./storage/'+OUTPUT_DIRECTORY, #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=100, # number of training epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=1000, # Number of update steps between two evaluations.
    save_steps=10000, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    )
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

device = "cuda:0"
model = model.to(device)
torch.cuda.empty_cache()

trainer.train()

model.save_pretrained(f"./trained_model/"+MODEL_NAME)
tokenizer.save_pretrained(f"./trained_model/"+MODEL_NAME)


test = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

out = []

for i in range(10):
    out.append(test('', num_return_sequences=10, max_length=240))

for set in out:
    for item in set:
        if len(item['generated_text']) > 20:
            if df.text.apply(lambda x: item['generated_text'] in x).sum() == 0:
                print(item['generated_text'])
                print()



out = []

for i in range(5):
    out.append(test('Why', num_return_sequences=10, max_length=240))

for set in out:
    for item in set:
        if len(item['generated_text']) > 20:
            print(item['generated_text'])
            print()


# Load Previous Point
model = AutoModelForCausalLM.from_pretrained('./trained_model/'+MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained('./trained_model/'+MODEL_NAME)
device = "cuda:0"
model = model.to(device)