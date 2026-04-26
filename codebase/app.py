import streamlit as st
from rag_graph import graph, config

st.set_page_config(page_title="TechSupport RAG", page_icon="🛠️")

st.title("🛠️ TechSupport RAG Assistant")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about your TechNova X1000 router..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching manual..."):
        # Invoke the graph
        result = graph.invoke(
            {"question": prompt, "messages": st.session_state.messages}, 
            config
        )
    
    answer = result.get("answer", "I encountered an error processing your request.")
    escalate = result.get("escalate", False)
    sources = result.get("context", [])

    if escalate:
        st.warning("⚠️ This request requires human intervention.")
        with st.expander("Human Agent Panel"):
            human_reply = st.text_area("Response for the customer:", placeholder="Enter manual resolution...")
            if st.button("Send Response"):
                answer = f"**[Human Agent]:** {human_reply}"
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()
    else:
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander("View Sources"):
                    for i, source in enumerate(sources):
                        st.info(f"Source {i+1}: {source}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})