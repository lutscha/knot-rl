from dataclasses import dataclass
from enum import Enum
import random
import json

from typing import List

random.seed(42)

def inv_perm(perm: list[int]) -> list[int]:
    res = [0 for i in range(len(perm))]
    for i, x in enumerate(perm):
        res[x] = i
    return res

@dataclass
class BraidGenerator:
    i: int
    sign: int

    def __str__(self):
        return f"{self.i}{'+' if self.sign == 1 else '-'}"

class Braid:
    n_strands: int
    repr: list[BraidGenerator]

    def __init__(self, n_strands: int, repr: list[BraidGenerator]):
        self.n_strands = n_strands
        self.repr = repr

    def to_perm(self) -> list[int]:
        perm = [i for i in range(self.n_strands)]
        for generator in self.repr:
            i = generator.i
            perm[i], perm[i - 1] = perm[i - 1], perm[i]
        return perm

    def __str__(self):
        return " ".join(str(generator) for generator in self.repr)

    def __mul__(self, other: 'Braid') -> 'Braid':
        return Braid(max(self.n_strands, other.n_strands), self.repr + other.repr)

class Orientation(Enum):
    POS = 0
    NEG = 1

class VisitType(Enum):
    OVER = 0
    UNDER = 1

def inv(visit_type: VisitType) -> VisitType:
    return VisitType.OVER if visit_type == VisitType.UNDER else VisitType.UNDER


@dataclass
class Visit:
    index: int
    type: VisitType
    orientation: Orientation

class Link:
    n_components: int
    n_crossings: int

    n_conn_visits: list[int]
    visits: list[Visit]


    def __init__(self, n_components: int, n_crossings: int, n_conn_visits: list[int], visits: list[Visit]):
        self.n_components = n_components
        self.n_crossings = n_crossings
        self.n_conn_visits = n_conn_visits
        self.visits = visits

    def __str__(self):
        return f"Link(n_components={self.n_components}, n_crossings={self.n_crossings}, n_conn_visits={self.n_conn_visits}, visits={self.visits})"
    

def braid_to_link(braid: Braid) -> Link:
    visits = [[] for i in range(braid.n_strands)]
    cnt = [0 for i in range(braid.n_strands)]

    def push(strand, other_strand, type, sign):
        visits[strand].append((other_strand, cnt[other_strand], type, sign)) # 

    
    perm = [i for i in range(braid.n_strands)]


    for k, generator in enumerate(braid.repr):
        i = generator.i
        strand_left, strand_right = perm[i-1], perm[i]

        type = VisitType.OVER if generator.sign == 1 else VisitType.UNDER
        sign = Orientation.POS if generator.sign == 1 else Orientation.NEG

        push(strand_left, strand_right, inv(type), sign)
        push(strand_right, strand_left, type, sign)
        cnt[strand_left] += 1
        cnt[strand_right] += 1
        perm[i], perm[i - 1] = perm[i - 1], perm[i]

    first_ind = [-1 for i in range(braid.n_strands)]

    visits_seen = 0
    cur_comp = 0

    inv_perm_ = inv_perm(perm)

    n_conn_visits = []
    for strand in range(braid.n_strands):
        if first_ind[strand] != -1:
            continue

        cur_comp += 1
        n_conn_visits.append(0)

        cur = strand
        while True:
            first_ind[cur] = visits_seen
            visits_seen += cnt[cur]
            cur = inv_perm_[cur]

            n_conn_visits[-1] += cnt[cur]
            if cur == strand:
                break

    visits_result = [None for i in range(2 * len(braid.repr))]

    for strand, strand_visits in enumerate(visits):
        pref = first_ind[strand]
        for ind, entry in enumerate(strand_visits):
            other_strand, other_ind, type_, sign = entry
            visits_result[pref + ind] = Visit(other_ind + first_ind[other_strand], type_, sign)

    assert visits_seen == 2 * len(braid.repr)

    return Link(n_components=cur_comp, n_crossings=len(braid.repr), n_conn_visits=n_conn_visits, visits=visits_result)

def complete_perm_to_long_cycle(beta: List[int]) -> List[int]:
    """
    Given beta (0-based permutation of length n), find a shortest word pi
    in the generators sigma_1,...,sigma_{n-1} (adjacent transpositions)
    such that beta * pi is a single n-cycle.

    Returns:
        gens: list of generator indices [i1, ..., ik], meaning
              pi = sigma_{i1} sigma_{i2} ... sigma_{ik},
              where sigma_i = (i-1, i) as a transposition on {0,...,n-1}.
        The length k is minimal and equals (#cycles(beta) - 1).
    """
    n = len(beta)
    beta = beta[:]  # work on a copy
    gens: List[int] = []

    while True:
        # Step 1: compute cycle id of each element under current beta
        cid = [-1] * n       # cid[x] = which cycle x belongs to
        num_cycles = 0
        for start in range(n):
            if cid[start] != -1:
                continue
            # new cycle starting at 'start'
            x = start
            while cid[x] == -1:
                cid[x] = num_cycles
                x = beta[x]
            num_cycles += 1

        # if already one cycle, we are done
        if num_cycles == 1:
            break

        # Step 2: find adjacent i, i+1 in DIFFERENT cycles and apply sigma_{i+1}
        for i in range(n - 1):
            if cid[i] != cid[i + 1]:
                # use generator sigma_{i+1} (i.e. transposition (i, i+1))
                gens.append(i + 1)            # 1-based index for sigma
                beta[i], beta[i + 1] = beta[i + 1], beta[i]  # beta := beta * (i i+1)
                break
        else:
            # Theoretically this can’t happen: if num_cycles > 1 there must
            # be some boundary between cycles along the line 0,1,...,n-1.
            raise RuntimeError("No merging adjacent pair found — should be impossible.")

    return gens


def random_signs(n) -> list[int]:
    return [random.choice([1, -1]) for _ in range(n)]

def zip_to_braid(gens, signs, n_strands) -> Braid:
    assert len(gens) == len(signs)
    repr = [BraidGenerator(gens[i], signs[i]) for i in range(len(gens))]
    return Braid(n_strands, repr)

def random_transitive_braid(n_strands, n_gens) -> Braid:
    gens = [random.randint(1, n_strands - 1) for _ in range(n_gens)]
    signs = random_signs(n_gens)

    braid = zip_to_braid(gens, signs, n_strands)

    completion_gens = complete_perm_to_long_cycle(braid.to_perm())
    completion_signs = random_signs(len(completion_gens))
    completion_braid = zip_to_braid(completion_gens, completion_signs, n_strands)

    res = braid * completion_braid

    return res


def random_knot(n_strands, n_crossings) -> Link:
    braid = random_transitive_braid(n_strands, n_crossings)
    return braid_to_link(braid)


SIGN_SHIFT = 0
R3_SHIFT = 3
N_R3_MOVES = 4
R4_SHIFT = R3_SHIFT + N_R3_MOVES      # 7
N_R4_MOVES = 4
TYPE_SHIFT = R4_SHIFT + N_R4_MOVES    # 11

def visit_flags(v) -> int:
    sign_bit = 0 if v.orientation == Orientation.POS else 1
    type_bit = 0 if v.type == VisitType.OVER else 1
    return (sign_bit << SIGN_SHIFT) | (type_bit << TYPE_SHIFT)

def serialize_link(link) -> str:
    return json.dumps(
        {
            "n_components": link.n_components,
            "n_crossings": link.n_crossings,
            "n_conn_visits": link.n_conn_visits,
            # each entry: [mate, flags]
            "visits": [
                [v.index, visit_flags(v)]
                for v in link.visits
            ],
        },
        indent=2,
    )

if __name__ == "__main__":
    link = random_knot(2, 100)

    with open("knot_data/dumb_knot.json", "w") as f:
        f.write(serialize_link(link))
