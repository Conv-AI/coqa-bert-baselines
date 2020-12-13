from coqa_bert_inference import *
embedding_length = 36454 
bertModel = ModelLoader (
        model_name='BERT', model_path='./output/output4004/best/model.pth', device='cuda', save_state_dir="./output/output4004/")
data = {
    "context": "Corona is the name of a virus group. These viruses are already known to India. SARS disease found in 2003 or MARS disease found in 2012 are also diseases caused by corona virus. But the corona virus that was detected in the outbreak in Wuhan, China, in December 2019, is different from the previous corona virus. Therefore, it is called Novel, the new corona virus. The World Health Organization has named the disease Covid-19. Corona virus is passed on from Animals to humans. It is found mainly in the bats. Due to innumerable deforestation, increasing urbanization, raw meat eating habits, etc., microorganisms in the animal world enter the human body. Symptoms of the disease are mainly associated with the respiratory system. They are generally similar to influenza illness. Symptoms include cold, cough, shortness of breath, fever, pneumonia, sometimes kidney failure.",
    "questions": [
        "What is Corona?"
        
    ],
    "answers": [
        
    ]
}
bertModel.getResult(data)
