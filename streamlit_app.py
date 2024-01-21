# import altair as alt
# import numpy as np
# import pandas as pd
# import streamlit as st

# """
# # Welcome to Streamlit!

# Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
# If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
# forums](https://discuss.streamlit.io).

# In the meantime, below is an example of what you can do with just a few lines of code:
# """

# num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
# num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

# indices = np.linspace(0, 1, num_points)
# theta = 2 * np.pi * num_turns * indices
# radius = indices

# x = radius * np.cos(theta)
# y = radius * np.sin(theta)

# df = pd.DataFrame({
#     "x": x,
#     "y": y,
#     "idx": indices,
#     "rand": np.random.randn(num_points),
# })

# st.altair_chart(alt.Chart(df, height=700, width=700)
#     .mark_point(filled=True)
#     .encode(
#         x=alt.X("x", axis=None),
#         y=alt.Y("y", axis=None),
#         color=alt.Color("idx", legend=None, scale=alt.Scale()),
#         size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
#     ))



import streamlit as st
from pydantic import BaseModel
import streamlit_pydantic as sp
from typing import Optional
from datetime import date

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

class CustomerData(BaseModel):
    customer_group_1: str
    customer_group_2: str
    title: str
    first_name: str
    last_name: str
    street: str
    postal_code: str
    city: str
    email: Optional[str]
    country: str
    language: str
    birth_date: Optional[date]
    age: Optional[int]
    nl_online: Optional[str]
    fb_nl: Optional[str]
    number_of_orders: int
    net_revenue: float
    total_revenue: float
    interests_preferences: Optional[str]
    style: Optional[str]
    shirt: int
    top: int
    pants: int
    dress: int
    shoes: int
    accessories: int

data = sp.pydantic_form(key="my_form", model=CustomerData)
if data:
    st.json(data.json())