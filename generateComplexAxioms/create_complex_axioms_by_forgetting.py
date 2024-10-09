import mowl

mowl.init_jvm("2g")

from tqdm import tqdm
import random
import os
from mowl.owlapi import OWLAPIAdapter, OWLSubClassOfAxiom
from owlready2 import get_ontology
from mowl.datasets.base import PathDataset
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import OWLClass, OWLObjectProperty, OWLObjectSomeValuesFrom, OWLObjectIntersectionOf, \
    OWLSubClassOfAxiom
from axiom2dict import transfer_axioms


def is_syntactic_subclass(C, C1) -> bool:
    """
    Recursively checks if class C is a syntactic subclass of C1. This function is based on the role depth:

    - If role depth of C is 0, checks if C1 is C ∩ C'
    - Otherwise, decomposes C and C1 and checks the components recursively.

    :param C: The candidate subclass (OWLClass or complex expression)
    :param C1: The candidate superclass (OWLClass or complex expression)
    :return: True if C is a subclass of C1 syntactically, False otherwise
    """
    # Case 1: If role depth is 0, check if C1 contains C as a conjunction
    if isinstance(C1, OWLClass):
        # Check if the atomic concept C is one of the conjuncts in C1
        if isinstance(C1, OWLObjectIntersectionOf):
            # C is a conjunct in C1
            return any(C == conjunct for conjunct in C1.getOperands())
        elif isinstance(C1, OWLClass):
            # Direct comparison
            return C == C1

    # Case 2: For complex classes (role depth > 0)
    elif isinstance(C, OWLObjectIntersectionOf):
        # Recursively check all conjuncts in the intersection
        for conjunct in C.getOperands():
            if not is_syntactic_subclass(conjunct, C1):
                return False
        return True

    elif isinstance(C, OWLObjectSomeValuesFrom):
        # For existential restrictions (∃ r.D1)
        role_C = C.getProperty()  # Get the role (r) in ∃ r.D1
        filler_C = C.getFiller()  # Get the filler (D1)

        if isinstance(C1, OWLObjectIntersectionOf):
            # Find all components of form ∃ r.D' in C1
            for conjunct in C1.getOperands():
                if isinstance(conjunct, OWLObjectSomeValuesFrom) and conjunct.getProperty() == role_C:
                    filler_C1 = conjunct.getFiller()  # Get D' from ∃ r.D'
                    # Recursively check if D1 is a subclass of D'
                    if is_syntactic_subclass(filler_C, filler_C1):
                        return True
        elif isinstance(C1, OWLObjectSomeValuesFrom):
            # Directly check if the role and filler match between C and C1
            role_C1 = C1.getProperty()  # Get the role in ∃ r.D'
            filler_C1 = C1.getFiller()  # Get the filler D'
            return role_C == role_C1 and is_syntactic_subclass(filler_C, filler_C1)

    return False  # Default case if nothing matches


def eliminate_redundant_axioms(axioms):
    """
    Eliminate redundant axioms from the ontology based on syntactic subclass checking.

    :param ontology: The mOWL Ontology object
    :return: A list of non-redundant axioms
    """
    non_redundant_axioms = []
    redundant_axioms = set([])

    for axiom in tqdm(axioms):
        if isinstance(axiom, OWLSubClassOfAxiom):
            C = axiom.getSubClass()  # The left side of the axiom
            D = axiom.getSuperClass()  # The right side of the axiom

            redundant = False

            # Check against all other axioms for redundancy
            for other_axiom in axioms:
                if other_axiom in redundant_axioms:
                    continue
                if other_axiom != axiom and isinstance(other_axiom, OWLSubClassOfAxiom):
                    C_other = other_axiom.getSubClass()
                    D_other = other_axiom.getSuperClass()

                    # Check if C_other is a syntactic subclass of C
                    if is_syntactic_subclass(C_other, C):
                        # Check if D is a subclass of D_other
                        if is_syntactic_subclass(D, D_other):
                            redundant = True
                            break

            if not redundant:
                non_redundant_axioms.append(axiom)  # Keep this axiom if not redundant
            else:
                redundant_axioms.add(axiom)  # Discard this axiom if redundant

    return non_redundant_axioms


# Check if the axiom is EL-compliant
def is_el_axiom(axiom):
    # EL allows only subclass axioms with atomic classes, conjunctions, and existential restrictions
    if isinstance(axiom, OWLSubClassOfAxiom):
        sub_class = axiom.getSubClass()
        super_class = axiom.getSuperClass()

        # Ensure both subclass and superclass are EL-compliant
        return is_el_expression(sub_class) and is_el_expression(super_class)
    else:
        # Reject non-subclass axioms
        return False


# Check if a class expression is EL-compliant
def is_el_expression(expression):
    if isinstance(expression, OWLClass):
        # Atomic class or top concept is allowed
        return True
    elif isinstance(expression, OWLObjectSomeValuesFrom):
        # Existential restriction is allowed (∃r.C)
        filler = expression.getFiller()
        return is_el_expression(filler)  # Check the filler (C)
    elif isinstance(expression, OWLObjectIntersectionOf):
        # Conjunction is allowed (A ⊓ B)
        return all(is_el_expression(expr) for expr in expression.getOperands())
    else:
        # Any other construct (disjunction, negation, universal quantification, etc.) is not allowed in EL
        return False


# Function to filter non-EL axioms from the ontology
def filter_non_el_axioms(axioms):
    el_axioms = []

    # Iterate over all axioms in the ontology
    for axiom in axioms:
        if is_el_axiom(axiom):
            el_axioms.append(axiom)

    return el_axioms


# Load ontology using mowl OWLAPIAdapter
def load_ontology(path):
    ontology = PathDataset(path).ontology
    return ontology


# Get all concept names from the ontology
def get_concept_names(ontology):
    classes = ontology.getClassesInSignature(Imports.fromBoolean(True))
    return [str(cls.toString()).replace("<", "").replace('>', "") for cls in classes]


# Save the ontology to a file to pass it to LETHE
def transfer_ontology(ontology_path, result_path):
    # transfer the ontology to standard OWL FSS format by:
    # 1 . add "Prefix" and "Ontology(" to the ontology
    # 2 . add ")" to the end of the ontology
    with open(ontology_path, 'r') as f:
        ontology = f.readlines()

    with open(result_path, 'w') as f:
        f.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n\n")
        f.write("Ontology(\n")
        for line in ontology:
            f.write(line)
        f.write(")\n")


# Forget concept names using LETHE
def perform_forgetting(ontology_file, path_sig):
    forget_command = [
        './forget.sh', '--owlFile', ontology_file, '--signature', path_sig
    ]
    print('forgetting:')
    os.system(' '.join(forget_command))
    print('done forgetting')


# Function to count the number of concepts (classes) and roles (object properties) in an axiom
def count_concepts_and_roles(axiom):
    # return number of "<" appearances in the axiom
    return str(axiom.toString()).count("<")


# Function to filter axioms by their length (number of concepts and roles)
def filter_axioms_by_length(ontology, k_min, k_max):
    filtered_axioms = []

    # Iterate over all axioms in the ontology
    for axiom in ontology.getTBoxAxioms(Imports.fromBoolean(True)):
        # Count the number of concepts and roles in the axiom
        length = count_concepts_and_roles(axiom)

        # Keep only axioms with length >= k
        if length >= k_min and length <= k_max:
            filtered_axioms.append(axiom)

    return filtered_axioms


def one_turn_forgetting(ontology, temp_ontology_file, file_path):
    # Step 2: Randomly select 50% of concept names
    all_concepts = get_concept_names(ontology)
    num_to_forget = 1000
    random_concepts = random.sample(all_concepts, num_to_forget)

    # Step 3: Save the random concept names to a file
    path_sig = 'temp_signature.txt'
    with open(path_sig, 'w') as f:
        f.write('\n'.join(random_concepts))

    # Step 4: Perform forgetting with LETHE
    perform_forgetting(temp_ontology_file, path_sig)

    # Step 5: Load the modified ontology back
    modified_ontology = load_ontology("result.owl")

    # Step 6: Output axioms longer than 4
    filtered_axioms = filter_axioms_by_length(modified_ontology, 4, 10)
    filtered_axioms = filter_non_el_axioms(filtered_axioms)

    # save the filtered axioms
    ont_name = file_path.split("/")[-1].split(".")[0]
    with open(f"{ont_name}_filtered_axioms.txt", "a") as f:
        for axiom in filtered_axioms:
            print(axiom.toString())
            f.write(str(axiom.toString()) + '\n')

    return filtered_axioms


# Main process
def main(file_path, concept2index_path, relation2index_path):
    # Step 0: Save the ontology and concept as signature for use with LETHE
    temp_ontology_file = 'temp_ontology.owl'
    transfer_ontology(file_path, temp_ontology_file)

    # Step 1: Load the ontology
    ontology = load_ontology(temp_ontology_file)

    filtered_axioms = []
    num_turns = 0
    while len(filtered_axioms) < 1000:
        num_turns += 1
        filtered_axioms += one_turn_forgetting(ontology, temp_ontology_file, file_path)

    print('eliminating redundant axioms:')
    if len(filtered_axioms) > 1000:
        filtered_axioms = random.sample(filtered_axioms, 1000)
    filtered_axioms = eliminate_redundant_axioms(filtered_axioms)
    print(f"Number of turns: {num_turns}")
    print(f"Number of axioms: {len(filtered_axioms)}")

    output_path = ont_name + "_filtered_axioms_dict.txt"
    transfer_axioms(filtered_axioms, concept2index_path, relation2index_path, output_path)


# File paths to your ontology and the output file
ont_name = 'name of Ontology'

input_ontology_file = ' path of Ontology'
concept2index_path = " path of classes2id.json"
relation2index_path = " path of relations2id.json"

if __name__ == "__main__":
    main(input_ontology_file, concept2index_path, relation2index_path)
