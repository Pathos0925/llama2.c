// Bounded scratchpad — small models need strict limits or they waste the
// context budget on rambling. Keep these in sync with World.setTask /
// updateNotes / setGoal.
export const LIMITS = {
  goal:  120,
  task:  100,
  notes: 500,
};

export function clip(field, s) {
  const max = LIMITS[field];
  if (!max) return s || '';
  return (s || '').toString().slice(0, max);
}
