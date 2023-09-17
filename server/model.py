import pickle as pkl
import pandas as pd
import sys
import cohere
import config
co = cohere.Client(config.Cohere_API_KEY) # This is your trial API key
with open('hackathon_model_file.pkl', 'rb') as pickle_file:
    content = pkl.load(pickle_file)
pivot_book = pd.read_csv('pivot_book.csv')
pivot_book['title'] = pivot_book['title'].str.lower()
pivot_book = pivot_book.set_index('title')
def main(prompt):
    input = pivot_book.iloc[pivot_book.index == prompt.strip().lower(), :].values.reshape(1,-1) 
    #print(book_name.strip()=='Harry Potter and the Chamber of Secrets (Book 2)')
    if len(input[0]) == 0:
        response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=300,
        temperature=0.9,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=[],
        return_likelihoods='NONE')
        return response.generations[0].text
        #input = pivot_book.iloc[pivot_book.index == 'No Safe Place'.lower(), :].values.reshape(1,-1)
    distances , suggestions = content.kneighbors(input)
    #for i in range(len(suggestions)):
    #    print(pivot_book.index[suggestions[i]])
    return ', and '.join(pivot_book.index[suggestions][0].tolist()[1:])
#print(main('Harry Potter and the Chamber of Secrets (Book 2)')[0])


if __name__ == "__main__":
    print(main(sys.argv[1]))
    #print(main('Harry Potter and the Chamber of Secrets (Book 2)'))