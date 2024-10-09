from mowl.owlapi import OWLAPIAdapter
import json
from org.semanticweb.owlapi.model import OWLClass, OWLObjectSomeValuesFrom, OWLObjectIntersectionOf, OWLObjectUnionOf, \
    OWLObjectComplementOf, OWLSubClassOfAxiom

# Decompose OWL Axiom into an index-based dictionary using concept2index and relation2index
# Decompose OWL Axiom into an index-based dictionary using concept2index and relation2index
def decompose_axiom_with_indices(concept, concept2index, relation2index):
    if isinstance(concept, OWLClass):
        # Atomic class: replace with concept index
        return concept2index.get(f"<{str(concept.toStringID())}>", f"Unknown concept: {concept.toStringID()}")

    elif isinstance(concept, OWLObjectSomeValuesFrom):
        # Existential restriction (∃r.C)
        property_name = relation2index.get(f"<{str(concept.getProperty().toStringID())}>",
                                           f"Unknown relation: {concept.getProperty().toStringID()}")
        filler = decompose_axiom_with_indices(concept.getFiller(), concept2index, relation2index)
        return {"exists": [property_name, filler]}

    elif isinstance(concept, OWLObjectIntersectionOf):
        # Conjunction (A ⊓ B)
        conjuncts = [decompose_axiom_with_indices(expr, concept2index, relation2index) for expr in concept.getOperands()]
        return {"conjunct": conjuncts}

    else:
        raise ValueError("Unknown axiom type")


# Process SubClassOf axioms and decompose them
def process_subclass_axiom(subclass_axiom, concept2index, relation2index):
    # Get the subclass and superclass
    subclass = subclass_axiom.getSubClass()
    superclass = subclass_axiom.getSuperClass()

    # Decompose both subclass and superclass
    subclass_decomposed = decompose_axiom_with_indices(subclass, concept2index, relation2index)
    superclass_decomposed = decompose_axiom_with_indices(superclass, concept2index, relation2index)

    # Return as a dictionary for easy manipulation
    return {"subclass": subclass_decomposed, "superclass": superclass_decomposed}


# Function to filter and process only SubClassOf axioms
def filter_and_process_subclass_axioms(axioms, concept2index, relation2index):
    processed_axioms = []

    # Iterate over all axioms in the ontology
    for axiom in axioms:
        # Check if the axiom is a SubClassOf axiom
        if isinstance(axiom, OWLSubClassOfAxiom):
            processed_axioms.append(
                process_subclass_axiom(axiom, concept2index, relation2index)
            )

    return processed_axioms



# Main function to load ontology, load dictionaries, and process axioms
def transfer_axioms(axioms, concept2index_path, relation2index_path, output_path):
    # Load the concept2index and relation2index dictionaries
    with open(concept2index_path, 'r') as f:
        concept2index = eval(
            f.read())  # Make sure this is a safe file (you may use safer parsing methods like json or pickle)

    with open(relation2index_path, 'r') as f:
        relation2index = eval(f.read())  # Make sure this is a safe file

    # Process axioms with indices
    decomposed_axioms = filter_and_process_subclass_axioms(axioms, concept2index, relation2index)

    # save the decomposed axioms
    with open(output_path, 'w') as f:
        for axiom in decomposed_axioms:
            f.write(json.dumps(axiom) + '\n')


