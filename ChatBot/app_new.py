import streamlit as st
from streamlit_chat import message as msg

st.snow()

st.audio('winter-day-christmas-holidays-270802.mp3', format="audio/wav", start_time=0, sample_rate=None, end_time=None, loop=True, autoplay=True)
st.title('Welcome to North Pole, Chat with Santa!')



##### Program Logic ####

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

def call_model(state: MessagesState):
    llm, prompt_template = get_configurations()
    prompt = prompt_template.invoke(state)
    response = llm.invoke(prompt)
    return {"messages": response}

def get_configurations():
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are Santa Claus, the jolly and magical figure of Christmas. 
            Stay in character as a cheerful, wise, and warm-hearted Santa, spreading joy and kindness.
            Tone: Use a friendly, festive tone with phrases like “Ho ho ho!” and “Merry Christmas!”
            Personality: Be kind, patient, and magical. Reference Santa’s world (North Pole, elves, reindeer).
            Roleplay: Respond with imagination and charm, staying true to Santa’s magical persona.
            Wishes: Listen to Christmas wishes, offer hope and positivity, but avoid making promises.
            Engagement: Share jokes, carols, traditions, or answer questions about Santa’s life.
            Inclusivity: Respect diverse traditions and spread universal holiday cheer.
            You will answer questions a Santa could answer, if the questions are out of the boundary of Santa, gracefully say that you dont know."""),
                MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return llm,prompt_template

def get_langgraph():
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def get_santa_response(query):
    # Very first time, setting the config
    if 'app' not in st.session_state:
        st.session_state['app'] = get_langgraph()
        st.session_state['config'] = {"configurable": {"thread_id": "abc345"}}
    app = st.session_state['app']
    config = st.session_state['config']
    input_messages = [HumanMessage(query)]
    #output = app.invoke({"messages": input_messages}, config)
    app.invoke({"messages": input_messages}, config)
    chat_history = app.get_state(config).values["messages"]
    return chat_history

if __name__ == "__main__":
    
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    
    with st.form(key='user_form', clear_on_submit=True):
        user_input = st.text_input(label='Your Message goes here...', placeholder="Type and Hit enter to get hold of Santa...", key='input')
        submit_button = st.form_submit_button(label='Chat')
        
        if submit_button and user_input:
            with st.spinner('Getting message from Santa...'):
                    chat_history = get_santa_response(user_input)
                    
                    # writing the chat history using streamlit chat message format
            
                    for message in chat_history:
                        if message.type == 'human':
                            #msg(message.content, is_user=True, avatar_style=st.image('santa-claus.png'))
                            msg(message.content, is_user=True, avatar_style=':material/featured_seasonal_and_gifts:')
                        else:
                            #msg(message.content, avatar_style=st.image('santa-claus.png'))
                            msg(message.content, avatar_style=':material/featured_seasonal_and_gifts:')
            
                
            
            


