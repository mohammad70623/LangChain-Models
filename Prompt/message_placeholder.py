from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

#Chat template 
chat_template = ChatPromptTemplate([
    ('system', 'you are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

#Load Chat History 
chat_history=[]
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
print(chat_history)

#create prompt 
prompt =chat_template.invoke({
    'chat_history': chat_history,
    'query':'where is my friend?'
})

print(prompt)