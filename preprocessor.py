import spacy
import pandas as pd
import os

class Preprocessor:
    def __init__(self, language='en_core_web_sm'):
        self.nlp = spacy.load(language, disable=["parser", "ner"])


    
    def pipeline(self, text):

        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if token.is_stop:
                continue
            elif token.text == "<PARTY>":
                tokens.append("party")
            elif token.text == "<p>":
                tokens.append("paraphrase")
            else:
                tokens.append(token.lemma_)
        return " ".join(tokens)
    
    
    def preprocess_files(self, input_folder, output_file):

        processed_data_list = []

        for file_name in os.listdir(input_folder):
            file_path = os.path.join(input_folder, file_name)
            try:
                data = pd.read_csv(file_path, sep = "\t")
                if 'text_en' in data.columns:
                    data['preprocessed_text'] = data['text_en'].apply(self.pipeline)
                else:
                    data['preprocessed_text'] = data['text'].apply(self.pipeline)

                processed_data_list.append(data)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
            

        processed_data = pd.concat(processed_data_list, ignore_index=True)
        processed_data.to_csv(output_file, sep="\t", index=False)


#proc = Preprocessor()
#proc.preprocess_files("data/orientation", "data/lemmatized_data.tsv")