import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# let's create a main function to receive two phisopger names and a topic iniatiation from the command line
def pholosphers_say():
    parser = argparse.ArgumentParser()
    parser.add_argument('--philosopher1', type=str, default='Epictetus')
    parser.add_argument('--philosopher2', type=str, default='Socrates')
    parser.add_argument('--topic', type=str, default='philosophy')
    args = parser.parse_args()
    philosopher1 = args.philosopher1
    philosopher2 = args.philosopher2
    topic = args.topic
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    inputs = tokenizer.encode(f'{philosopher1}: According to my opinion regarding the {topic},', return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, do_sample=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    inputs = tokenizer.encode(f'{philosopher2} what I believe regarding the notion of {topic} is ', return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, do_sample=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# add a main function to receive a sequence and a length from teh command line
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=str, default='Epictetus was a Stoic philosopher. He was born in the Greek city of Hierapolis, Phrygia, in 55 AD. His father was a physic "'
    )
    parser.add_argument('--length', type=int, default=200)
    args = parser.parse_args()
    sequence = args.sequence
    length = args.length
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    inputs = tokenizer.encode(sequence, return_tensors='pt')
    outputs = model.generate(inputs, max_length=length, do_sample=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()