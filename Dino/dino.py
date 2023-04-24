from logic import *

ATruthoraptor = Symbol("A is a Truthoraptor")
ALieosaurus = Symbol("A is a Lieosaurus")

BTruthoraptor = Symbol("B is a Truthoraptor")
BLieosaurus = Symbol("B is a Lieosaurus")

CTruthoraptor = Symbol("C is a Truthoraptor")
CLieosaurus = Symbol("C is a Lieosaurus")

# Puzzle 0
# A says "I am both a Truthoraptor and a Lieosaurus."
knowledge0 = And(
    # Knowledge Base
    Or(ATruthoraptor, ALieosaurus),
    Not(And(ATruthoraptor, ALieosaurus)),

    # If A is telling the truth, A is both truthoraptor and lieosaurus
    Implication(ATruthoraptor, And(ATruthoraptor, ALieosaurus)),

    # If A is not telling the truth, A is not both truthoraptor and lieosaurus
    Implication(ALieosaurus, Not(And(ATruthoraptor, ALieosaurus)))
)

# Puzzle 1
# A says "We are both Lieosauruss."
# B says nothing.
knowledge1 = And(
    # Knowledge Base
    Or(ATruthoraptor, ALieosaurus),
    Or(BTruthoraptor, BLieosaurus),
    Not(And(ATruthoraptor, ALieosaurus)),
    Not(And(BTruthoraptor, BLieosaurus)),

    # If A is telling the truth, both A and B are lieosaurus
    Implication(ATruthoraptor, And(ALieosaurus, BLieosaurus)),

    # If A is lying, the statement A is lieosaurus and B is lieosaurus is false
    Implication(ALieosaurus, Not(And(ALieosaurus, BLieosaurus)))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    Or(ATruthoraptor, ALieosaurus),
    Not(And(ATruthoraptor, ALieosaurus)),
    Or(BTruthoraptor, BLieosaurus),
    Not(And(BTruthoraptor, BLieosaurus)),

    # A says they're the same kind
    # First implication is if A is telling truth, second implication is if A is lying
    Implication(ATruthoraptor, Or(And(ATruthoraptor, BTruthoraptor), And(ALieosaurus, BLieosaurus))),
    Implication(ALieosaurus, Not(Or(And(ATruthoraptor, BTruthoraptor), And(ALieosaurus, BLieosaurus)))),

    # B says they're NOT the same kind
    # First implication is if B is telling truth, second implication is if B is lying
    Implication(BTruthoraptor, Or(And(BTruthoraptor, ALieosaurus), And(BLieosaurus, ATruthoraptor))),
    Implication(BLieosaurus, Not(Or(And(BTruthoraptor, ALieosaurus), And(BLieosaurus, ATruthoraptor))))
)

# Puzzle 3
# A says either "I am a Truthoraptor." or "I am a Lieosaurus.", but you don't know which.
# B says "A said 'I am a Lieosaurus'."
# B says "C is a Lieosaurus."
# C says "A is a Truthoraptor."
knowledge3 = And(
    Or(ATruthoraptor, ALieosaurus), 
    Not(And(ATruthoraptor, ALieosaurus)),
    Or(BTruthoraptor, BLieosaurus), 
    Not(And(BTruthoraptor, BLieosaurus)),
    Or(CTruthoraptor, CLieosaurus), 
    Not(And(CTruthoraptor, CLieosaurus)),
    
    #if A is a Truthoraptor, A can be a truthoraptor or a lieosaurus 
    #If A is a ALieosaurus, then it can't be a Truthoraptor and a Lieosaurus.
    Implication(ATruthoraptor,Or(ATruthoraptor,ALieosaurus)),
    Implication(ALieosaurus,Not(Or(ATruthoraptor,ALieosaurus))),
    
   
    #if B is a Truthoraptor than C is a Lieasaurus
    #if B is a Lieosaurus than C can't be a Lieosaurus 
    Implication(BTruthoraptor,CLieosaurus),
    Implication(BLieosaurus,Not(CLieosaurus)),
    
    
    #if C is a truthoraptor than A is a truthoraptor
    #if C is a liesaurus then A is not a Truthoraptor
    Implication(CTruthoraptor,ATruthoraptor),
    Implication(CLieosaurus,Not(ATruthoraptor))
)

def main():
    symbols = [ATruthoraptor, ALieosaurus, BTruthoraptor, BLieosaurus, CTruthoraptor, CLieosaurus]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()