import streamlit as st

# Define the SessionState class to manage state across pages
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Main function to define the Streamlit app
def main():
    #st.title("Multi-Page Streamlit App")

    # Initialize SessionState
    session_state = SessionState(page=None)

    # Define the content for each page
    page_contents = {
        "Page 1": "This is the content for Page 1.",
        "Page 2": "This is the content for Page 2.",
        "Page 3": "This is the content for Page 3."
    }

    # Sidebar to select pages
    selected_page = st.sidebar.button("Select Page", list(page_contents.keys()))

    # Display content based on selected page
    if selected_page:
        session_state.page = selected_page
        st.write(page_contents[selected_page])

    # Highlighted sentences to navigate to different pages
    if session_state.page != "Page 1":
        if st.button("Go to Page 1"):
            session_state.page = "Page 1"
            st.experimental_rerun()

    if session_state.page != "Page 2":
        if st.button("Go to Page 2"):
            session_state.page = "Page 2"
            st.experimental_rerun()

    if session_state.page != "Page 3":
        if st.button("Go to Page 3"):
            session_state.page = "Page 3"
            st.experimental_rerun()

# Run the Streamlit app
if __name__ == "__main__":
    main()
