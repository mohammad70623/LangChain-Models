from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# Load model
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=-1,
    pipeline_kwargs={
        "temperature": 0.01,
        "max_new_tokens": 512
    }
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> Detailed Report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2nd prompt -> Summary
template2 = PromptTemplate(
    template="Write a 5 line summary of the following text:\n{text}",
    input_variables=["text"]
)

# Invoke 1st prompt
prompt1 = template1.invoke({"topic": "Daffodil International University"})
result1 = model.invoke(prompt1)

# Invoke 2nd prompt
prompt2 = template2.invoke({"text": result1.content})
result2 = model.invoke(prompt2)

print(result1.content)
print(result2.content)
