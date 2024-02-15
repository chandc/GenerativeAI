import ipywidgets as widgets

def slider(description, d_value, min_value, max_value, step): 
    return widgets.IntSlider( 
     value=d_value, 
     min=min_value, 
     max=max_value, 
     step=step, 
     description=description  )

def text(title, placeholder): 
    return widgets.Text(
           placeholder=placeholder,
           description=title)


def textML(title, placeholder): 
    return widgets.Textarea(
           placeholder=placeholder,
           description=title, layout=widgets.Layout(width='70%', height='80px'))


def sel(options:list,default:int,rows:int,description:str):
    return widgets.Select(
            options=options,
            value=default,
            rows=rows,
            description=description )

def RB(options:list, value:int,description:str):
    return widgets.RadioButtons(
    options=options,
    value=value,
#    value='pineapple', # Defaults to 'pineapple'
#    layout={'width': 'max-content'}, # If the items' names are long
    description=description,
    disabled=False
)
