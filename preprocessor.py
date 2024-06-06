import spacy
import pandas as pd
import os

from tqdm import tqdm

class Preprocessor:
    def __init__(self, language='en_core_web_sm'):
        self.nlp = spacy.load(language, disable=["parser", "ner"])
    
    def pipeline(self, text):
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if token.is_stop:
                continue
            else:
                tokens.append(token.lemma_)
        return " ".join(tokens)
    
    
    def preprocess_files(self, input_folder, output_dir, subtask):
        processed_data_list = []
        data_list = []

        for file_name in tqdm(os.listdir(input_folder)):

            file_path = os.path.join(input_folder, file_name)
            try:
                data = pd.read_csv(file_path, sep = "\t")
                if 'text_en' in data.columns:
                    data['preprocessed_text'] = data['text_en'].apply(self.pipeline)
                    data_list.append(data[['text_en', 'label']].rename(columns={'text_en': 'text'}))
                else:
                    data['preprocessed_text'] = data['text'].apply(self.pipeline)
                    data_list.append(data[['text', 'label']])

                processed_data_list.append(data[['preprocessed_text','label']].rename(columns={'preprocessed_text': 'text'}))
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
            
            out_path = os.path.join(output_dir, 'out', file_name)
            data_list[-1].to_csv(out_path, index=False, sep='\t')
            
            out_path_processed = os.path.join(output_dir, 'out', f'stopword{file_name}')
            processed_data_list[-1].to_csv(out_path_processed, index=False, sep='\t')

        #processed_data = pd.concat(processed_data_list, ignore_index=True)
        #processed_data.to_csv(os.path.join(output_dir, f'{subtask}_lemmatized_data.tsv'), sep="\t", index=False)

        #unprocessed_data = pd.concat(data_list, ignore_index=True)  
        #unprocessed_data.to_csv(os.path.join(output_dir, f'{subtask}_data.tsv'))

if __name__=='__main__':
    proc = Preprocessor()
    subtask = 'orientation'
    proc.preprocess_files(os.path.join("data/", subtask), "data/", subtask)