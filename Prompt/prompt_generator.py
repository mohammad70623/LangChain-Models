from langchain_core.prompts import PromptTemplate 

#Template 
template = PromptTemplate(
    template="""
    Please summarize the research paper titled "{paper_input}" with the following specifications: 
    Explanation Style: {style_input}
    Explanation Length: {length_input}
    1.Mathematical Details:
     - Include relevant equations and mathematical equations if present in the paper.
     - Explain the mathematical concepts using simple , Intuitive code snippets where applicable.
    2.Analogies:
     - use relatable analogies to simplify complex idea.
    If certain information is not available in the paper, response with: "Insufficient Information available" instead of guessing.
    Ensure the summary is clear, accurate and aligned with the provided style and length.
""", 
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True
)

template.save('template.json')