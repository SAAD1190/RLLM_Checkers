################################################################################
################################ Structered Output ################################
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class Moves_Checkers(BaseModel):
    Moves : str = Field(..., description="A list of moves in the form of ((start_x, start_y), (end_x, end_y))")
   

system_Checkers = """
You are an AI agent tasked with recommending the top 3 optimal moves in a 10x10 checkers game.  
Your recommendations are based on analyzing the board state and prioritizing moves that maximize strategic advantage for the current player.  

### Board Representation  

- The board is represented as a 10x10 grid.  
- Each cell has a value:  
  - 0: Empty cell  
  - 1: White pawn  
  - 2: White queen  
  - -1: Black pawn  
  - -2: Black queen  

### Move Representation 
 
- ((start_x, start_y), (end_x, end_y))  

### Your Output  
- You will generate exactly **3 recommended moves** in Python list format for the current player.  
- Each move should adhere to the format of the action structure described above.  

### Evaluation Criteria  

- Prioritize capturing opponent pieces whenever possible.  
- Promote pawns to queens if a move to the farthest row is possible.  
- Avoid moves that expose pieces to immediate capture.  
- Maximize control of strategic areas on the board.  

### Input Structure  

1. Current board state as a 10x10 grid.  
2. Current player's turn (1 for White, -1 for Black).  

### Output Example  

For a 10x10 board state provided as input:  

[[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
 [ 0, 0, 0, 1, 0, -1, 0, 0, 0, 0],  
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  
 [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]  

And the current player is `1 (White)`.  
Your response should be a Python list in this format:  

[  
    ((4, 5), (3, 6)),  
    ((6, 1), (7, 0)),  
    ((5, 2), (3, 4))  
]  

### Rules for Interaction  

- Do not include explanations or commentary in your response.  
- Indexing between 0 and 9.

### Objective 
 
Your goal is to assist in narrowing the action space for an RL agent that selects the final move from your recommendations.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_Checkers),
        ("human", "The board state here :\n {Board_State} \n The current player is {Current_Player}  - Indexing between 0 and 9."),
    ]
)

from langchain_openai import ChatOpenAI
import pandas as pd
import time


RETRIES = 5

LLM = ChatOpenAI(
    model="qwen2.5:14b",
    #temperature=0,
    api_key='ollama',
    base_url="http://localhost:11434/v1",
)
#LLM_Structered = LLM.with_structured_output(Moves_Checkers)
Suggester = prompt | LLM 


def Run_LLM(Board_State, Current_Player):
    #print("\n LLM is running.... \n")
    Input = {"Board_State":Board_State, "Current_Player":Current_Player}
    for i in range(RETRIES):
        #print(f"Invocation {i+1}/{RETRIES}...\n")
        try:
            response = Suggester.invoke(Input)
            # Parsing
            print(response.content) 
            suggestions = eval(response.content)
            return suggestions
        except Exception as e:
            pass
    return [((4, 5), (3, 6)), ((6, 1), (7, 0)), ((5, 2), (3, 4))]



### Quick Test
"""
board_state = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

player = "1"
"""
#response = Run_LLM(board_state,player)
#print(response)
