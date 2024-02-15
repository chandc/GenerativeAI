This notebook conducts the following tasks to assist a potential buyer in finding a home that will meet a set of prescribed preferences:

1. loads a real estate listing from a CSV file, 
2. creates embeddings for the property description of each listing and stores them in the Chroma vectorstore for query,
3. collects home and neighborhood preferences from a potential buyer,
4. uses the retriever utility function in LangChain to locate a set of listings that best match the user-provided preferences in home size and budget,
5. uses GPT-3-turbo-0125 language model to provide a summary description for each home that resonates with the potential buyer.


```python
import json
import pandas as pd

from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import DataFrameLoader

from langchain.docstore.document import Document


import ipywidgets as widgets
from widgets import slider, textML, RB

import os
from dotenv import load_dotenv, find_dotenv

```


```python
#
# look up API key from the .env file
#
_ = load_dotenv(find_dotenv()) # read local .env file

```


```python
#
# load the listing database from a csv file
#

df = pd.read_csv("./home_listing.csv")

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Listing Number</th>
      <th>Neighborhood</th>
      <th>Price</th>
      <th>Bedrooms</th>
      <th>Bathrooms</th>
      <th>House Size</th>
      <th>House Description</th>
      <th>Neighborhood Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>L000001</td>
      <td>West Lake Village</td>
      <td>$800,000</td>
      <td>3</td>
      <td>2.0</td>
      <td>2,500 sqft</td>
      <td>Welcome to this charming home nestled in the h...</td>
      <td>This home is situated in a quiet and friendly ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L000002</td>
      <td>Agoura Hills</td>
      <td>$900,000</td>
      <td>3</td>
      <td>2.5</td>
      <td>2,200 sqft</td>
      <td>This stunning home in Agoura Hills offers a pe...</td>
      <td>Located in the desirable Agoura Hills communit...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>L000003</td>
      <td>Newbury Park</td>
      <td>$850,000</td>
      <td>4</td>
      <td>2.5</td>
      <td>2,100 sqft</td>
      <td>Welcome to this beautifully updated home in Ne...</td>
      <td>This home is located in a friendly and family-...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>L000004</td>
      <td>Oak Park</td>
      <td>$750,000</td>
      <td>3</td>
      <td>2.0</td>
      <td>2,100 sqft</td>
      <td>This charming home in Oak Park offers a perfec...</td>
      <td>Located in a peaceful and picturesque neighbor...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>L000005</td>
      <td>Dos Vientos</td>
      <td>$950,000</td>
      <td>4</td>
      <td>3.0</td>
      <td>2,400 sqft</td>
      <td>This stunning home in Dos Vientos offers luxur...</td>
      <td>Located in the highly desirable Dos Vientos co...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f"There are {df.shape[0]} listings")
```

    There are 70 listings



```python
df.shape
```




    (70, 8)




```python
#
# remove duplicate records
#

df = df.drop_duplicates(subset=['Listing Number'])

```


```python
df.shape
```




    (51, 8)




```python
df["Price"] = df.apply(lambda row: int(row["Price"][1:].replace(",","")), axis=1)
```


```python
df["Features"] =  df.apply(
                lambda row: Document(page_content=row["House Description"]+ ' ' + row["Neighborhood Description"], 
                                     metadata={"LN": row["Listing Number"], "Price": row["Price"], "Size": row["House Size"],
                                              "Bedrooms": row["Bedrooms"], "Bathrooms": row["Bathrooms"]
                                              } ), axis=1
          )
```


```python
df.head()
    
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Listing Number</th>
      <th>Neighborhood</th>
      <th>Price</th>
      <th>Bedrooms</th>
      <th>Bathrooms</th>
      <th>House Size</th>
      <th>House Description</th>
      <th>Neighborhood Description</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>L000001</td>
      <td>West Lake Village</td>
      <td>800000</td>
      <td>3</td>
      <td>2.0</td>
      <td>2,500 sqft</td>
      <td>Welcome to this charming home nestled in the h...</td>
      <td>This home is situated in a quiet and friendly ...</td>
      <td>page_content="Welcome to this charming home ne...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>L000002</td>
      <td>Agoura Hills</td>
      <td>900000</td>
      <td>3</td>
      <td>2.5</td>
      <td>2,200 sqft</td>
      <td>This stunning home in Agoura Hills offers a pe...</td>
      <td>Located in the desirable Agoura Hills communit...</td>
      <td>page_content="This stunning home in Agoura Hil...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>L000003</td>
      <td>Newbury Park</td>
      <td>850000</td>
      <td>4</td>
      <td>2.5</td>
      <td>2,100 sqft</td>
      <td>Welcome to this beautifully updated home in Ne...</td>
      <td>This home is located in a friendly and family-...</td>
      <td>page_content="Welcome to this beautifully upda...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>L000004</td>
      <td>Oak Park</td>
      <td>750000</td>
      <td>3</td>
      <td>2.0</td>
      <td>2,100 sqft</td>
      <td>This charming home in Oak Park offers a perfec...</td>
      <td>Located in a peaceful and picturesque neighbor...</td>
      <td>page_content="This charming home in Oak Park o...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>L000005</td>
      <td>Dos Vientos</td>
      <td>950000</td>
      <td>4</td>
      <td>3.0</td>
      <td>2,400 sqft</td>
      <td>This stunning home in Dos Vientos offers luxur...</td>
      <td>Located in the highly desirable Dos Vientos co...</td>
      <td>page_content="This stunning home in Dos Viento...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df.dtypes
```


```python
#
# set up the chat prompt template with system instructions
#
template = """You are a professional real estate agent assisting home buyers. 
Use the retrieved Listings below to identify which ones can best match the given Preferences. 
You can select up to 3 items from the Listings for each Answer. 
The Answer should start in a new line with the message: 
"## Thank you for your interest, home(s) that best meet your preferences are: ##"
Each offered item must start in a separate line with all the metadata that include "LN", "Price", "Size", "Bedrooms", "Bathrooms" ** no exceptions **. 
They are then followed by a tailored description of the listing that resonates with buyer's preferences, 
try to subtly emphasize aspects of the property that align with what the buyer is looking for, however, ** they MUST be factual and you cannot make things up **.
You must strictly adhere to these instructions. Do not provide any other information not asked for.
Preferences: {question} 
Listings: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)
```

    input_variables=['context', 'question'] messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template='You are a professional real estate agent assisting home buyers. \nUse the retrieved Listings below to identify which ones can best match the given Preferences. \nYou can select up to 3 items from the Listings for each Answer. \nThe Answer should start in a new line with the message: \n"## Thank you for your interest, home(s) that best meet your preferences are: ##"\nEach offered item must start in a separate line with all the metadata that include "LN", "Price", "Size", "Bedrooms", "Bathrooms" ** no exceptions **. \nThey are then followed by a tailored description of the listing that resonates with buyer\'s preferences, \ntry to subtly emphasize aspects of the property that align with what the buyer is looking for, however, ** they MUST be factual and you cannot make things up **.\nYou must strictly adhere to these instructions. Do not provide any other information not asked for.\nPreferences: {question} \nListings: {context} \nAnswer:\n'))]



```python
#
# set up widgets to collect user input iteractively
#
preferences = textML("preferences:","e.g. house amenities, style, neighborhood and transportation")
budget = slider("Budget in $:", 800000, 700000, 1000000, 50000)
bedroom = RB([2,3,4,5],3, "number of bedrooms:")
bathroom = RB([2,2.5,3],2, "number of bathrooms:")
rooms = widgets.HBox([bedroom, bathroom])

box = widgets.VBox([budget, rooms, preferences])
display(box)
```


    VBox(children=(IntSlider(value=800000, description='Budget in $:', max=1000000, min=700000, step=50000), HBox(…



```python
print(f"Budget is ${budget.value}")
print(f"Minimum number of bedrooms is {bedroom.value}")
print(f"Minimum number of bathrooms is {bathroom.value}")
print(f"Preferences listed: {preferences.value}")

```

    Budget is $1000000
    Minimum number of bedrooms is 3
    Minimum number of bathrooms is 2.5
    Preferences listed: a gourmet kitchen, large family room with lots of windows, fenced backyard with a BBQ grill, near top-rated schools and recreational areas, lots of tree and away from city center, low crime rate.
    



```python
d_list = df[ (df["Price"] <= budget.value) & (df["Bedrooms"] >= bedroom.value) & (df["Bathrooms"] >= bathroom.value) ]["Features"] 
```


```python
data = []
for d in d_list:
    data.append(d)
    
```


```python
print(f"{len(d_list)} on the shortlist")
```

    11 on the shortlist



```python
question = f"Find me home listings with {preferences.value}"

print(question)
```

    Find me home listings with a gourmet kitchen, large family room with lots of windows, fenced backyard with a BBQ grill, near top-rated schools and recreational areas, lots of tree and away from city center, low crime rate.
    



```python
if (len(data)==0): print("** sorry! no listing in the inventory will meet your perferences, please modify your inputs **")
```


```python
#
# set up embeddings and Chroma vectorstore
#
docsearch = Chroma.from_documents(data, OpenAIEmbeddings())
retriever = docsearch.as_retriever()
```


```python
docs = docsearch.similarity_search_with_score(question)
```


```python
docs
```




    [(Document(page_content="Nestled in the highly sought-after Oak Park community, this stunning home offers the perfect blend of luxury and comfort. The open and bright floor plan features a gourmet kitchen with granite countertops, high-end appliances, and a large island. The spacious living and dining area feature soaring ceilings, a cozy fireplace, and large windows that look out onto the backyard. The master suite is a true oasis, with a luxurious en-suite bathroom and a private balcony overlooking the mountains. The backyard features a built-in BBQ, multiple seating areas, and plenty of space for outdoor entertaining. This home also offers a three-car garage and solar panels. Oak Park is known for its beautiful parks, top-rated schools, and close-knit community. Residents can enjoy miles of hiking and biking trails, as well as nearby shopping and dining options. The neighborhood also offers easy access to major highways and is just a short drive from nearby beaches and the city. Don't miss out on the opportunity to live in this charming community!", metadata={'LN': 'L056789', 'Price': 900000, 'Size': '2,500 sqft', 'Bedrooms': 4, 'Bathrooms': 3.0}),
      0.27267828583717346),
     (Document(page_content="This beautiful home is nestled in the quiet and friendly city of Moorpark. The interior features an open floor plan with high ceilings and plenty of natural light. The kitchen boasts granite countertops, stainless steel appliances, and a breakfast bar. The spacious master suite includes a walk-in closet and a luxurious en-suite bathroom with a soaking tub and separate shower. The backyard is perfect for entertaining, with a covered patio, built-in BBQ, and fire pit. This home also has a three-car garage for your convenience. Located in the heart of Moorpark, this home offers easy access to a variety of shopping and dining options. The neighborhood is known for its top-rated schools and friendly community. Outdoor enthusiasts will love the nearby parks, hiking trails, and golf courses. With easy access to the 23 freeway, commuting to nearby cities is a breeze. Don't miss out on the opportunity to live in one of the most desirable areas of Moorpark.", metadata={'LN': 'L038634', 'Price': 950000, 'Size': '2,200 sqft', 'Bedrooms': 4, 'Bathrooms': 3.0}),
      0.29052677750587463),
     (Document(page_content="This stunning home is located in the desirable community of Moorpark. The spacious living room features high ceilings, a cozy fireplace, and large windows that provide plenty of natural light. The kitchen boasts granite countertops, stainless steel appliances, and a sunny breakfast nook. Upstairs, you'll find the luxurious master suite with a walk-in closet and an en-suite bathroom with a spa tub. The backyard is an entertainer's dream, with a built-in BBQ, a covered patio, and a sparkling pool. This home also includes a two-car garage and a laundry room. Located in a peaceful and family-friendly neighborhood, this home is just minutes away from the bustling Moorpark Town Center, where you can find shopping, dining, and entertainment options galore. The nearby parks and hiking trails offer the perfect escape for outdoor enthusiasts. This home is also in a top-rated school district and offers easy access to major freeways, making it a convenient location for commuters. Don't miss your chance to live in the desirable Moorpark community!", metadata={'LN': 'L567890', 'Price': 950000, 'Size': '2,400 sqft', 'Bedrooms': 3, 'Bathrooms': 2.5}),
      0.2922130227088928),
     (Document(page_content="Welcome to this beautifully updated home in the desirable community of Calabasas. The spacious living room features high ceilings, a cozy fireplace, and large windows that provide plenty of natural light. The kitchen boasts granite countertops, stainless steel appliances, and a breakfast nook with a bay window. Upstairs, you'll find the luxurious master suite with a walk-in closet and an en-suite bathroom with a soaking tub. The backyard is an entertainer's dream, with a built-in BBQ, a fire pit, and a sparkling pool. This home also includes a two-car garage and a separate laundry room. Located in a prestigious and exclusive gated community, this home offers the perfect blend of luxury and privacy. You'll have access to top-rated schools, beautiful parks, and upscale shopping and dining options, all just a short drive away. The nearby freeway allows for an easy commute to the city, and the stunning views of the surrounding mountains make coming home feel like a vacation. Don't miss your chance to live in this highly sought-after neighborhood in Calabasas!", metadata={'LN': 'L345678', 'Price': 1000000, 'Size': '2,200 sqft', 'Bedrooms': 4, 'Bathrooms': 2.5}),
      0.2947937250137329)]




```python
#
# set up LLM and RAG chain
#

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.5)


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


```


```python
question = f"Find me home listings with {preferences.value}"

print(question)
```

    Find me home listings with a gourmet kitchen, large family room with lots of windows, fenced backyard with a BBQ grill, near top-rated schools and recreational areas, lots of tree and away from city center, low crime rate.
    


### Top 3 listings that meet buyer's preferences 


```python
output = rag_chain.invoke(question)
```


```python
print(output)
```

    ## Thank you for your interest, home(s) that best meet your preferences are: ##
    LN: L056789
    Price: 900000
    Size: 2,500 sqft
    Bedrooms: 4
    Bathrooms: 3.0
    This stunning home in Oak Park features a gourmet kitchen with granite countertops, high-end appliances, and a large island. The spacious living and dining area have large windows that provide lots of natural light. The backyard includes a built-in BBQ, multiple seating areas, and ample space for outdoor entertaining. Located near top-rated schools and recreational areas, this home offers a peaceful setting with plenty of trees away from the city center.
    
    LN: L038634
    Price: 950000
    Size: 2,200 sqft
    Bedrooms: 4
    Bathrooms: 3.0
    Nestled in Moorpark, this beautiful home boasts a gourmet kitchen with granite countertops and stainless steel appliances. The large family room has high ceilings and plenty of natural light. The fenced backyard features a BBQ grill and a fire pit, perfect for outdoor gatherings. Situated near top-rated schools and recreational areas, this home offers a low crime rate and a serene environment.
    
    LN: L345678
    Price: 1000000
    Size: 2,200 sqft
    Bedrooms: 4
    Bathrooms: 2.5
    Located in Calabasas, this beautifully updated home offers a gourmet kitchen with granite countertops and stainless steel appliances. The spacious living room has large windows that bring in natural light. The fenced backyard includes a BBQ grill and a fire pit for outdoor entertainment. Situated near top-rated schools and recreational areas, this home provides a tranquil setting with a low crime rate.



```python

```


```python

```


```python

```


```python

```
