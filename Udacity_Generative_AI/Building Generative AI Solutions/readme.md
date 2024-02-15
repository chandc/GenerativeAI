### Introduction of Project HomeMatch

Project HomeMatch transforms standard real estate listings into personalized narratives that align with the unique preferences and needs of potential buyers.

Home buyers can input their requirements and preferences, such as home size, budget, amenities, and lifestyle choices, into HomeMatch to discover potential matches in the listings. HomeMatch interprets these inputs in natural language and deciphers a buyer's requests beyond basic keyword searches.

Using a Large Language Model (LLM), HomeMatch stores the descriptions of all available property listings as vector embeddings in a vector store. These embeddings allow the vector store, equipped with built-in algorithms, to identify listings with the closest semantic match to the buyer's preferences.

In this case, we choose to present the top three matches. Moreover, HomeMatch utilizes an LLM to rewrite the home description in a manner that emphasizes aspects most relevant to the buyer's preferences. It ensures personalization by highlighting appealing characteristics without altering factual information about the property.

To test out this design concept, we implemented the following 5-step process in the `HomeMatch.ipynb` Jupyter notebook:

1. Load a real estate listing from the `./home_listing.csv` file.
2. Create embeddings for the property description of each listing and store them in the Chroma vector store for querying.
3. Collect home and neighborhood preferences from a potential buyer using an interactive interface.
4. Use the retriever utility function in LangChain to locate a set of listings that best match the user-provided preferences in home size and budget.
5. Utilize OpenAI's GPT-3-turbo-0125 LLM to provide a summary description for the top 3 matches in the listing that resonates with the potential buyer.

The `HomeMatch.ipynb` Jupyter notebook is built with a kernel that consists of the following libraries:

- pandas == 1.35
- langchain == 0.1
- ipwidgets == 8.1.1
- python-dotenv == 1.0.0

It is also accompanied by a set of helper functions in `widgets.py` to generate a slider, two radio buttons, and a text block to collect user input through an interactive interface. The `Generate_listing.ipynb` notebook synthetically generates 50 listings using LLM and a one-shot example, then stores them in `./home_listing.csv` as an input to `HomeMatch.ipynb`.

Sample outputs are shown in `output_1.md` and `output_2.md`.
