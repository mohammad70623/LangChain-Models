from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

#Chat template 
chat_template = ChatPromptTemplate([
    ('system', 'you are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])