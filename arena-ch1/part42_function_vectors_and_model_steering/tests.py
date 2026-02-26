import string
from pathlib import Path

import numpy as np
import torch as t
from nnterp import StandardizedTransformer

N_LAYERS = 48
N_HEADS = 25
D_MODEL = 1600
D_HEAD = D_MODEL // N_HEADS

# root should be the section directory
section_dir = Path(__file__).parent


class ICLSequence:
    def __init__(self, word_pairs: list[list[str]]):
        self.word_pairs = word_pairs
        self.x, self.y = zip(*word_pairs)

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx: int):
        return self.word_pairs[idx]

    def prompt(self):
        """Returns the prompt, which contains all but the second element in the last word pair."""
        p = "\n\n".join([f"Q: {x}\nA: {y}" for x, y in self.word_pairs])
        return p[: -len(self.completion())]

    def completion(self):
        """Returns the second element in the last word pair (with padded space)."""
        return " " + self.y[-1]

    def __str__(self):
        """Prints a readable string representation of the prompt & completion (indep of template)."""
        return f"{', '.join([f'({x}, {y})' for x, y in self[:-1]])}, {self.x[-1]} ->".strip(", ")


class ICLDataset:
    def __init__(
        self,
        word_pairs: list[list[str]],
        size: int,
        n_prepended: int,
        bidirectional: bool = True,
        corrupted: bool = False,
        seed: int = 0,
    ):
        assert n_prepended + 1 <= len(
            word_pairs
        ), "Not enough antonym pairs in dataset to create prompt."

        self.word_pairs = word_pairs
        self.word_list = [word for word_pair in word_pairs for word in word_pair]
        self.size = size
        self.n_prepended = n_prepended
        self.bidirectional = bidirectional
        self.corrupted = corrupted
        self.seed = seed

        self.seqs = []
        self.prompts = []
        self.completions = []

        # Generate the dataset (by choosing random word pairs, and constructing `ICLSequence` objects)
        for n in range(size):
            np.random.seed(seed + n)
            random_pairs = np.random.choice(len(self.word_pairs), n_prepended + 1, replace=False)
            # Randomize the order of each word pair (x, y). If not bidirectional, we always have x -> y not y -> x
            random_orders = np.random.choice([1, -1], n_prepended + 1)
            if not (bidirectional):
                random_orders[:] = 1
            word_pairs = [
                self.word_pairs[pair][::order] for pair, order in zip(random_pairs, random_orders)
            ]
            # If corrupted, then replace y with a random word in all (x, y) pairs except the last one
            if corrupted:
                for i in range(len(word_pairs) - 1):
                    word_pairs[i][1] = np.random.choice(self.word_list)
            seq = ICLSequence(word_pairs)

            self.seqs.append(seq)
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())

    def create_corrupted_dataset(self):
        """Creates a corrupted version of the dataset (with same random seed)."""
        return ICLDataset(
            self.word_pairs,
            self.size,
            self.n_prepended,
            bidirectional=True,
            corrupted=False,
            seed=self.seed,
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return self.seqs[idx]


def test_calculate_h(calculate_h, model: StandardizedTransformer, solution: bool = False) -> None:
    """
    Tests the calculate_h function using a deterministic dataset of lowercase-uppercase letter pairs.

    Run with `solution=True` to save the vector we get from this function, rather than testing it
    against the previously saved vector.
    """
    # This same code was used to initially calculate the h vector, using `solution=True`
    word_pairs = list(zip(string.ascii_lowercase, string.ascii_uppercase))
    dataset = ICLDataset(word_pairs, size=5, n_prepended=5, bidirectional=False, seed=0)
    model_completions, h = calculate_h(model, dataset, layer=9)

    # Check model completions - GPT-2-XL may give different completions than GPT-J
    # so we just check the shape and that completions are returned
    assert len(model_completions) == 5, f"Expected 5 completions, got {len(model_completions)}"

    # Check shape
    assert h.shape == (D_MODEL,), f"Expected shape (d_model,), got {h.shape}"

    # Save new vector (for future tests) if solution=True
    if solution:
        t.save(h, section_dir / "data" / "test_h_gpt2xl.pt")
        print("Saved new h-vector.")
        return

    # Check h-vector against expected vector (which was saved using this same code)
    test_h_path = section_dir / "data" / "test_h_gpt2xl.pt"
    if test_h_path.exists():
        test_h = t.load(test_h_path, map_location="cpu", weights_only=True).float()
        mean_diff = (h.cpu().float() - test_h.cpu()).abs().mean().item()
        assert (
            mean_diff < 0.1
        ), f"Correct shape, but incorrect values: mean absolute diff = {mean_diff}"
    else:
        print("No saved h-vector found for GPT-2-XL. Saving one now...")
        t.save(h.cpu().float(), test_h_path)

    print("All tests in `test_calculate_h` passed.")


def test_intervene_with_h(intervene_with_h, model, h, ANTONYM_PAIRS) -> None:
    """
    Tests the intervene_with_h function by running it and checking that completions are returned.
    """
    import part42_function_vectors_and_model_steering.solutions as solutions

    # Get non-deterministic datasets
    zero_shot_dataset = ICLDataset(ANTONYM_PAIRS, size=5, n_prepended=0)

    # Run the intervene_with_h function, get the output
    print("Running your `intervene_with_h` function...")
    completions_zero_shot, completions_intervention = intervene_with_h(
        model, zero_shot_dataset, h, layer=9
    )
    print("Running `solutions.intervene_with_h` (so we can compare outputs) ...")
    completions_zero_shot_soln, completions_intervention_soln = solutions.intervene_with_h(
        model, zero_shot_dataset, h, layer=9
    )

    # Check each one is the same
    print("Comparing the outputs...")
    assert (
        completions_zero_shot == completions_zero_shot_soln
    ), "The model's zero-shot completions (without intervention) are not correct."
    assert (
        completions_intervention == completions_intervention_soln
    ), "The model's zero-shot completions (without intervention) are correct, but the model's zero-shot completions (when intervening with h) are not correct."

    print("\nAll tests in `test_intervene_with_h` passed.")


def test_calculate_fn_vector(calculate_fn_vector, model, solution: bool = False) -> None:
    # This same code was used to initially calculate the fn vector (everything is deterministic)
    word_pairs = list(zip(string.ascii_lowercase, string.ascii_uppercase))
    dataset = ICLDataset(word_pairs, size=3, n_prepended=2, seed=0)

    # First, test with just one attention head
    print("Testing for single head ... ")
    fn_vector = calculate_fn_vector(model, dataset, [(8, 1)])
    assert fn_vector.shape == (D_MODEL,), f"Expected shape (d_model,), got {fn_vector.shape}"

    if solution:
        t.save(fn_vector.cpu().float(), section_dir / "data" / "test_fn_vector_1_gpt2xl.pt")
        print("Saved new fn vector (single head).")
    else:
        fn_path = section_dir / "data" / "test_fn_vector_1_gpt2xl.pt"
        if fn_path.exists():
            fn_vector_expected = t.load(fn_path, weights_only=True)
            mean_diff = (fn_vector.cpu().float() - fn_vector_expected.cpu()).abs().mean().item()
            assert (
                mean_diff < 0.1
            ), f"Correct shape, but incorrect values: mean absolute diff = {mean_diff}"
        else:
            print("No saved fn vector found. Saving one now...")
            t.save(fn_vector.cpu().float(), fn_path)
    print("tests for single head passed.")

    # Next, test with more than one attention head
    print("Testing for multiple heads ... ")
    fn_vector = calculate_fn_vector(model, dataset, [(8, 1), (10, 1), (12, 1)])
    assert fn_vector.shape == (D_MODEL,), f"Expected shape (d_model,), got {fn_vector.shape}"

    if solution:
        t.save(fn_vector.cpu().float(), section_dir / "data" / "test_fn_vector_2_gpt2xl.pt")
        print("Saved new fn vector (multi head).")
    else:
        fn_path = section_dir / "data" / "test_fn_vector_2_gpt2xl.pt"
        if fn_path.exists():
            fn_vector_expected = t.load(fn_path, weights_only=True)
            mean_diff = (fn_vector.cpu().float() - fn_vector_expected.cpu()).abs().mean().item()
            assert (
                mean_diff < 0.1
            ), f"Correct shape, but incorrect values: mean absolute diff = {mean_diff}"
        else:
            print("No saved fn vector found. Saving one now...")
            t.save(fn_vector.cpu().float(), fn_path)
    print("tests for multiple heads passed.")

    print("\nAll tests in `test_calculate_fn_vector` passed.")
