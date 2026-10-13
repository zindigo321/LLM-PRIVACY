# Discrepancy Evidence: UID-20261012-v2 File Hash Mismatch

**Incident Time**: 2026-10-12T13:13:00+08:00  
**Discovery**: Pre-launch consistency check  
**Root Cause**: Post-freeze documentation edits during PREPARE

## Current State (Before Restoration)

### File Hashes

**uformer_blocks.py**:
- Expected (manifest): `3333f1862288e720d83aadbe8f171898f5f6943d2c0f58c9bd0132aba60b0747` (10,213 bytes)
- Actual: `9c68e47f375df7d000960f214452238eb8d717b2fb2f67b87f2feb072b8b8d6b` (10,225 bytes)
- Delta: +12 bytes

**uformer_ae.py**:
- Expected (manifest): `a3ad6e3f2b6e6f009158276f5534e2c68bf42781df18a1c1d9fcd0f5867c8606` (8,544 bytes)
- Actual: `4f14df26f3173c8a473b73890b1a871bc35ede9dba87711761482e81a176a01c` (8,533 bytes)
- Delta: -11 bytes

### Identified Edits

From conversation memory (Edit tool calls made after PREPARE report):

1. **uformer_blocks.py line 7-8**: Changed docstring reference
   - Original: `checks/test_upstream_equivalence.py`
   - Current: `checks/check_upstream_equivalence.py`
   - Expected delta: +1 byte (25→26 chars: "check_" vs "test_")

2. **uformer_ae.py line 9**: Changed docstring reference
   - Original: `checks/test_resnet_sampling_equivalence.py`
   - Current: `checks/check_resnet_sampling.py`
   - Expected delta: -19 bytes (45→26 chars)

### Byte Count Analysis

**uformer_blocks.py**: +12 bytes actual vs +1 byte expected from documented edit  
**Unaccounted**: +11 bytes

**uformer_ae.py**: -11 bytes actual vs -19 bytes expected from documented edit  
**Unaccounted**: +8 bytes

**Conclusion**: The available evidence does NOT fully explain the byte count changes. The conversation memory records only two specific filename substitutions, but the actual byte deltas suggest additional undocumented edits or encoding changes.

### Restoration Approach

1. Reverse the two documented edits exactly as stated in conversation memory
2. Verify resulting hashes against expected values
3. If hashes do not match, report the remaining discrepancy without further guessing

### Preserved Artifacts

- `incident_evidence_20261012/uformer_blocks.py.current` (SHA-256 `9c68e47f...b8d6b`)
- `incident_evidence_20261012/uformer_ae.py.current` (SHA-256 `4f14df26...6a01c`)
- `incident_evidence_20261012/manifest.json.frozen` (reference copy)

**Authorization**: Narrowly scoped restoration per approval dated 2026-10-12T13:15:00+08:00
