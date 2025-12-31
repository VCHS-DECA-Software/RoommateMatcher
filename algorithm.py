"""
Stable Roommate Matching Algorithm

This module implements a modified version of Irving's stable roommate algorithm
optimized for group assignments with configurable group sizes and constraints.
"""

import math
import random
import numpy as np
from typing import List, Optional, Set, Dict


class Group:
    """Represents a room group containing students."""

    def __init__(self, matching: 'Matching', members: Optional[List[int]] = None):
        """
        Initialize a group.

        Args:
            matching: The parent Matching instance
            members: Initial list of member indices (optional)
        """
        self.matching = matching
        self.members = members if members is not None else []
        self._temp_member = -1
        self._temp_score = -1.0
        self._cached_score: Optional[float] = None

    def add(self, member: int) -> None:
        """Add a member to the group."""
        self.members.append(member)
        self._cached_score = None

    def remove(self, member: int) -> None:
        """Remove a member from the group."""
        self.members.remove(member)
        self._cached_score = None

    def set_temp_candidate(self, member: int) -> float:
        """Set a temporary candidate for this group."""
        self._temp_member = member
        self._temp_score = self.matching.get_group_preference_for_member(
            member, self.members
        )
        return self._temp_score

    def confirm_temp_candidate(self) -> None:
        """Permanently add the temporary candidate to the group."""
        if self._temp_member >= 0:
            self.add(self._temp_member)
            self._temp_member = -1
            self._temp_score = -1.0

    def clear_temp_candidate(self) -> None:
        """Clear the temporary candidate without adding them."""
        self._temp_member = -1
        self._temp_score = -1.0

    def get_score(self) -> float:
        """Get cached group score, computing if necessary."""
        if self._cached_score is None:
            self._cached_score = self.matching.get_group_score(self.members)
        return self._cached_score

    @property
    def temp_member(self) -> int:
        return self._temp_member

    @property
    def temp_score(self) -> float:
        return self._temp_score

    def __len__(self) -> int:
        return len(self.members)


class Matching:
    """
    Stable roommate matching algorithm with group constraints.

    This implementation supports:
    - Configurable group sizes
    - Grade level constraints
    - Preference-based optimization with convergence detection
    """

    def __init__(
        self,
        prefs: np.ndarray,
        genders: Optional[np.ndarray] = None,
        grades: Optional[np.ndarray] = None,
        allowed_grade_diff: int = 1,
        group_size: int = 4,
        iter_count: int = 2,
        final_iter_count: int = 10,
    ):
        """
        Initialize the matching algorithm.

        Args:
            prefs: NxN preference matrix where prefs[i][j] = i's preference for j
            genders: Array of gender values for each student (optional)
            grades: Array of grade levels for each student (optional)
            allowed_grade_diff: Maximum allowed grade difference between roommates
            group_size: Maximum students per room
            iter_count: Optimization iterations during grouping phase
            final_iter_count: Max optimization iterations after all grouped
        """
        self.prefs = prefs
        self.genders = genders
        self.grades = grades
        self.allowed_grade_diff = allowed_grade_diff
        self.group_size = group_size
        self.iter_count = iter_count
        self.final_iter_count = final_iter_count

        self.num_members = prefs.shape[0]
        self.num_groups = math.ceil(self.num_members / group_size)

        # Use set for O(1) membership operations
        self.ungrouped: Set[int] = set(range(self.num_members))
        self.unfilled: List[Group] = []
        self.filled: List[Group] = []

        # Build index for O(1) ungrouped position lookup
        self._ungrouped_list: List[int] = list(range(self.num_members))

        self._initialize_seed_groups()

    def _initialize_seed_groups(self) -> None:
        """Create initial seed groups with random selection."""
        candidates = list(range(self.num_members))
        random.shuffle(candidates)

        for member in candidates:
            if len(self.unfilled) >= self.num_groups:
                break
            self.unfilled.append(Group(self, [member]))
            self.ungrouped.discard(member)

        self._ungrouped_list = list(self.ungrouped)

    def is_compatible(self, member1: int, member2: int) -> bool:
        """Check if two members are compatible based on constraints."""
        if self.genders is not None:
            if self.genders[member1] != self.genders[member2]:
                return False

        if self.grades is not None:
            grade_diff = abs(self.grades[member1] - self.grades[member2])
            if grade_diff > self.allowed_grade_diff:
                return False

        return True

    def can_join_group(self, member: int, group_members: List[int]) -> bool:
        """Check if a member can join a group."""
        return all(
            self.is_compatible(member, existing)
            for existing in group_members
        )

    def get_member_preference_for_group(self, member: int, group: List[int]) -> float:
        """Calculate a member's average preference for a group."""
        if not group:
            return 0.0
        total = sum(self.prefs[member][g] for g in group)
        return total / len(group)

    def get_group_preference_for_member(self, member: int, group: List[int]) -> float:
        """Calculate a group's average preference for a member."""
        if not group:
            return 0.0
        total = sum(self.prefs[g][member] for g in group)
        return total / len(group)

    def get_group_score(self, members: List[int]) -> float:
        """Calculate the internal preference score of a group."""
        if len(members) <= 1:
            return 0.0

        total = sum(
            self.prefs[i][j]
            for i in members
            for j in members
            if i != j
        )

        num_pairs = len(members) * (len(members) - 1)
        return total / num_pairs

    def get_total_score(self) -> float:
        """Calculate the average score across all filled groups."""
        if not self.filled:
            return 0.0

        total = sum(g.get_score() for g in self.filled)
        return total / len(self.filled)

    def solve(self) -> tuple:
        """Run the matching algorithm."""
        while self.ungrouped:
            self._assign_one_round()

        self.filled.extend(self.unfilled)
        self.unfilled = []

        self._optimize(use_filled=True)

        groups = [g.members for g in self.filled]
        return self.get_total_score(), groups

    def _assign_one_round(self) -> None:
        """Run one round of the assignment algorithm."""
        ungrouped_list = list(self.ungrouped)
        num_ungrouped = len(ungrouped_list)
        num_unfilled = len(self.unfilled)

        if num_ungrouped == 0:
            return

        # If no unfilled groups but members remain, handle them directly
        if num_unfilled == 0:
            self._handle_incompatible_members()
            self._update_filled_groups()
            return

        # Build index for O(1) lookup
        ungrouped_idx: Dict[int, int] = {m: i for i, m in enumerate(ungrouped_list)}

        proposed = np.zeros((num_ungrouped, num_unfilled), dtype=bool)
        is_assigned = [False] * num_ungrouped

        # Calculate preferences
        preferences = np.full((num_ungrouped, num_unfilled), -float('inf'))
        for i, member in enumerate(ungrouped_list):
            for j, group in enumerate(self.unfilled):
                if self.can_join_group(member, group.members):
                    preferences[i][j] = self.get_member_preference_for_group(
                        member, group.members
                    )

        pref_order = np.argsort(-preferences, axis=1)

        # Proposal-rejection loop with early exit
        while not all(is_assigned):
            made_progress = False

            for i, member in enumerate(ungrouped_list):
                if is_assigned[i]:
                    continue

                if proposed[i].all():
                    is_assigned[i] = True
                    made_progress = True
                    continue

                for j in pref_order[i]:
                    if proposed[i][j]:
                        continue

                    proposed[i][j] = True
                    group = self.unfilled[j]

                    if not self.can_join_group(member, group.members):
                        continue

                    score = self.get_group_preference_for_member(member, group.members)
                    if score > group.temp_score:
                        if group.temp_member >= 0:
                            prev_idx = ungrouped_idx[group.temp_member]
                            is_assigned[prev_idx] = False

                        group.set_temp_candidate(member)
                        is_assigned[i] = True
                        made_progress = True
                        break

            if not made_progress:
                break

        # Confirm assignments
        for group in self.unfilled:
            if group.temp_member >= 0:
                self.ungrouped.discard(group.temp_member)
                group.confirm_temp_candidate()

        self._optimize(use_filled=False)
        self._handle_incompatible_members()
        self._update_filled_groups()

    def _handle_incompatible_members(self) -> None:
        """Create new groups for members who can't join existing groups."""
        for member in list(self.ungrouped):
            can_join_any = any(
                self.can_join_group(member, g.members)
                for g in self.unfilled
            )

            if not can_join_any:
                self.unfilled.append(Group(self, [member]))
                self.ungrouped.discard(member)

    def _update_filled_groups(self) -> None:
        """Move full groups from unfilled to filled list."""
        newly_filled = [
            g for g in self.unfilled
            if len(g) >= self.group_size or not self.ungrouped
        ]

        for group in newly_filled:
            self.filled.append(group)
            self.unfilled.remove(group)

    def _optimize(self, use_filled: bool = True) -> None:
        """Optimize group assignments with convergence detection."""
        groups = self.filled if use_filled else self.unfilled
        max_iterations = self.final_iter_count if use_filled else self.iter_count

        for iteration in range(max_iterations):
            swaps_made = 0

            for i, group1 in enumerate(groups):
                for j, group2 in enumerate(groups):
                    if j <= i:
                        continue

                    # Try all member pairs between these two groups
                    for member1 in list(group1.members):
                        for member2 in list(group2.members):
                            if self._try_swap(group1, member1, group2, member2):
                                swaps_made += 1

            # Convergence: no swaps means we've reached a local optimum
            if swaps_made == 0:
                break

    def _try_swap(
        self,
        group1: Group,
        member1: int,
        group2: Group,
        member2: int
    ) -> bool:
        """Try swapping two members between groups if it improves scores."""
        # Safety check: ensure members are still in their respective groups
        # (they may have been swapped in a previous iteration)
        if member1 not in group1.members or member2 not in group2.members:
            return False

        # Check compatibility
        g1_others = [m for m in group1.members if m != member1]
        g2_others = [m for m in group2.members if m != member2]

        if not self.can_join_group(member2, g1_others):
            return False
        if not self.can_join_group(member1, g2_others):
            return False

        # Calculate score delta (more efficient than full recalc)
        old_score = group1.get_score() + group2.get_score()

        new_g1 = g1_others + [member2]
        new_g2 = g2_others + [member1]
        new_score = self.get_group_score(new_g1) + self.get_group_score(new_g2)

        if new_score > old_score:
            group1.remove(member1)
            group1.add(member2)
            group2.remove(member2)
            group2.add(member1)
            return True

        return False

    @staticmethod
    def from_csv(file_path: str, group_size: int = 4) -> 'Matching':
        """Create a Matching instance from a CSV preference matrix."""
        prefs = np.genfromtxt(file_path, delimiter=",")
        return Matching(prefs, group_size=group_size)
