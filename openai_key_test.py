import os

os.environ["OPENAI_API_KEY"] = "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"

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

def extract_sentences(dir_name='teext'):
    filenames = []
    for file in os.listdir(dir_name):
        filenames.append(file)
    print(filenames[0])
    doc_count = 1
    sentences = None
    data_folder = os.path.join(os.getcwd(), 'teext')
    for idx in range(doc_count):
        with open(os.path.join(data_folder, filenames[idx]), "r", encoding="utf-8") as fp:
            sentences = fp.read().split("\n")

    return sentences

if __name__ == "__main__":
    sentences = extract_sentences()
    
    trigger_template = """{sentence}

    Event trigger word: """

    topic_template = """{document} and these are trigger word list {triggers}

    What is the topic in few words: """


    trigger_prompt = PromptTemplate(template=trigger_template, input_variables=["sentence"])
    topic_prompt = PromptTemplate(template=topic_template, input_variables=["document", "triggers"])
    llm = OpenAI()

    llm_chain_trigger = LLMChain(prompt=trigger_prompt, llm=llm)
    llm_chain_topic = LLMChain(prompt=topic_prompt, llm=llm)

    # question = "Sam was accused of killing Rony at Amsterdam, but was later found to be innocent and framed for the murder."
    triggers = []
    for idx, sentence in enumerate(sentences):
        # print("Events in {}  are:\n".format(idx))
        triggers.append(llm_chain_trigger.run(sentence))
    
    # document = "".join(sentence for sentence in sentences)
    # print(llm_chain_topic.run({'document': document, 'triggers':triggers}))

    for idx, trigger in enumerate(triggers):
        print("\nsentence {} has trigger \n{}".format(idx, trigger))
