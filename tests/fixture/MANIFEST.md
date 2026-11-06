# Test fixture — manifest

Real images copied byte-for-byte from the supplied grape dataset by
`tests/build_fixture.py` (seed 42). Filenames are preserved exactly. Each file sits
in the same `{train,test}/Grape___<class>/` folder it occupies in the supplied split.
Regenerate with:

```bash
.venv/bin/python tests/build_fixture.py "/Users/bryce/Downloads/Literature Review paper/Grape/grape_dataset"
```

## Roles

| Role | Per class | What it exercises |
|---|---|---|
| `leak` | 2 leaves × 2 files | One leaf with its original in `test/` and an augmented copy in `train/`, and one the other way round. These are real cross-partition leaves from the supplied split, kept where they really are, so the audit must find them. |
| `group` | 3 leaves × 2 files | Original and augmented copy of one leaf, both in `train/`. Grouped splitting must keep each pair together. |
| `clean-test` | 1 leaf (not healthy) | Original in `test/` with no copy of that leaf in `train/` anywhere in the real data. Healthy has no such leaf, because its real test-leaf overlap is 100%. |
| `fill` | the rest, to 20 per class | Single originals, alternating `train/` and `test/`. Only the original is copied. |

## Deliberate leaks (8 leaves)

- `200b0ce7-899b-4cfe-b36b-1202ad34bc50` (Grape___Leaf_blight_(Isariopsis_Leaf_Spot)): `test/200b0ce7-899b-4cfe-b36b-1202ad34bc50___FAM_L.Blight 1617.JPG`; `train/200b0ce7-899b-4cfe-b36b-1202ad34bc50___FAM_L.Blight 1617_flipLR.JPG`
- `26c24a0b-568c-407b-90b6-7335ac631f8a` (Grape___Black_rot): `train/26c24a0b-568c-407b-90b6-7335ac631f8a___FAM_B.Rot 0536.JPG`; `test/26c24a0b-568c-407b-90b6-7335ac631f8a___FAM_B.Rot 0536_flipLR.JPG`
- `2c135c30-428c-4ca3-a86f-5aa1b8e32aa9` (Grape___Leaf_blight_(Isariopsis_Leaf_Spot)): `train/2c135c30-428c-4ca3-a86f-5aa1b8e32aa9___FAM_L.Blight 1573.JPG`; `test/2c135c30-428c-4ca3-a86f-5aa1b8e32aa9___FAM_L.Blight 1573_flipLR.JPG`
- `3473ab58-5217-4ab7-bfda-8d8683c5700d` (Grape___Esca_(Black_Measles)): `test/3473ab58-5217-4ab7-bfda-8d8683c5700d___FAM_B.Msls 4262.JPG`; `train/3473ab58-5217-4ab7-bfda-8d8683c5700d___FAM_B.Msls 4262_flipLR.JPG`
- `69a50f71-f8ee-4280-ac3e-c1d68e950c2e` (Grape___healthy): `train/69a50f71-f8ee-4280-ac3e-c1d68e950c2e___Mt.N.V_HL 6165.JPG`; `test/69a50f71-f8ee-4280-ac3e-c1d68e950c2e___Mt.N.V_HL 6165_180deg.JPG`
- `c3b72cd8-977b-4a6e-a833-7bcc87873882` (Grape___healthy): `test/c3b72cd8-977b-4a6e-a833-7bcc87873882___Mt.N.V_HL 9068.JPG`; `train/c3b72cd8-977b-4a6e-a833-7bcc87873882___Mt.N.V_HL 9068_180deg.JPG`
- `ca2d4db1-7716-4007-a5d9-fc6d23655c2b` (Grape___Black_rot): `test/ca2d4db1-7716-4007-a5d9-fc6d23655c2b___FAM_B.Rot 5051.JPG`; `train/ca2d4db1-7716-4007-a5d9-fc6d23655c2b___FAM_B.Rot 5051_flipLR.JPG`
- `db29b18d-163d-4566-81cb-1242ba4fb0c8` (Grape___Esca_(Black_Measles)): `train/db29b18d-163d-4566-81cb-1242ba4fb0c8___FAM_B.Msls 1012.JPG`; `test/db29b18d-163d-4566-81cb-1242ba4fb0c8___FAM_B.Msls 1012_flipLR.JPG`

## Expected audit on this fixture

Stage 2's audit must reproduce these exactly. They are also in `manifest.json`.

| Quantity | Value |
|---|---|
| images | 80 |
| source leaves | 60 |
| originals / augmented | 60 / 20 |
| train leaves / test leaves | 40 / 28 |
| test leaves also in train | 8 (28.6%) |
| test originals whose leaf is in train | 4 of 24 (16.7%) |

Per class, test leaves also in train:

- Grape___Black_rot: 2 of 7 (28.6%)
- Grape___Esca_(Black_Measles): 2 of 7 (28.6%)
- Grape___Leaf_blight_(Isariopsis_Leaf_Spot): 2 of 7 (28.6%)
- Grape___healthy: 2 of 7 (28.6%)

Total size: 1,363,973 bytes.

## Files

| Partition | Class | Role | Aug | Filename |
|---|---|---|---|---|
| test | Grape___Black_rot | leak | — | `ca2d4db1-7716-4007-a5d9-fc6d23655c2b___FAM_B.Rot 5051.JPG` |
| train | Grape___Black_rot | leak | _flipLR | `ca2d4db1-7716-4007-a5d9-fc6d23655c2b___FAM_B.Rot 5051_flipLR.JPG` |
| train | Grape___Black_rot | leak | — | `26c24a0b-568c-407b-90b6-7335ac631f8a___FAM_B.Rot 0536.JPG` |
| test | Grape___Black_rot | leak | _flipLR | `26c24a0b-568c-407b-90b6-7335ac631f8a___FAM_B.Rot 0536_flipLR.JPG` |
| train | Grape___Black_rot | group | — | `09cfe515-118c-475d-9f94-de565f56eb69___FAM_B.Rot 0360.JPG` |
| train | Grape___Black_rot | group | _flipLR | `09cfe515-118c-475d-9f94-de565f56eb69___FAM_B.Rot 0360_flipLR.JPG` |
| train | Grape___Black_rot | group | — | `61fff4d4-7012-4171-bfed-5ca53f55abdb___FAM_B.Rot 0352.JPG` |
| train | Grape___Black_rot | group | _flipLR | `61fff4d4-7012-4171-bfed-5ca53f55abdb___FAM_B.Rot 0352_flipLR.JPG` |
| train | Grape___Black_rot | group | — | `56fe18c8-e9bf-43ce-b7b8-960400a4555d___FAM_B.Rot 5092.JPG` |
| train | Grape___Black_rot | group | _flipLR | `56fe18c8-e9bf-43ce-b7b8-960400a4555d___FAM_B.Rot 5092_flipLR.JPG` |
| test | Grape___Black_rot | clean-test | — | `54f5ead3-b7f1-4320-914f-9dc9652c705a___FAM_B.Rot 0419.JPG` |
| train | Grape___Black_rot | fill | — | `25b99cd6-6e9d-48e9-a3f9-a4df821fb659___FAM_B.Rot 3271.JPG` |
| test | Grape___Black_rot | fill | — | `127b3861-2889-4bc6-93a6-60ec07c77612___FAM_B.Rot 3615.JPG` |
| train | Grape___Black_rot | fill | — | `cd7ac3e6-8980-4617-a736-d9fe011014c4___FAM_B.Rot 3373.JPG` |
| test | Grape___Black_rot | fill | — | `974fc349-46f5-49f6-b269-f4fcd75167a7___FAM_B.Rot 0637.JPG` |
| train | Grape___Black_rot | fill | — | `1cb9fa90-804f-4b63-9cf7-bab274ab9c5e___FAM_B.Rot 3325.JPG` |
| test | Grape___Black_rot | fill | — | `a30c651a-ef6a-4c84-9fc8-b79663d71a9a___FAM_B.Rot 0589.JPG` |
| train | Grape___Black_rot | fill | — | `bc29bf91-3597-48d1-8ec3-8fe9112185ca___FAM_B.Rot 0760.JPG` |
| test | Grape___Black_rot | fill | — | `ac051205-5c7e-4aa4-b2bd-393272c807cd___FAM_B.Rot 0676.JPG` |
| train | Grape___Black_rot | fill | — | `cf01898b-cac1-425f-b9e3-482b8f3e82b9___FAM_B.Rot 3176.JPG` |
| test | Grape___Esca_(Black_Measles) | leak | — | `3473ab58-5217-4ab7-bfda-8d8683c5700d___FAM_B.Msls 4262.JPG` |
| train | Grape___Esca_(Black_Measles) | leak | _flipLR | `3473ab58-5217-4ab7-bfda-8d8683c5700d___FAM_B.Msls 4262_flipLR.JPG` |
| train | Grape___Esca_(Black_Measles) | leak | — | `db29b18d-163d-4566-81cb-1242ba4fb0c8___FAM_B.Msls 1012.JPG` |
| test | Grape___Esca_(Black_Measles) | leak | _flipLR | `db29b18d-163d-4566-81cb-1242ba4fb0c8___FAM_B.Msls 1012_flipLR.JPG` |
| train | Grape___Esca_(Black_Measles) | group | — | `cb863d41-b4ad-4a93-b2b7-53270959a26e___FAM_B.Msls 1873.JPG` |
| train | Grape___Esca_(Black_Measles) | group | _flipLR | `cb863d41-b4ad-4a93-b2b7-53270959a26e___FAM_B.Msls 1873_flipLR.JPG` |
| train | Grape___Esca_(Black_Measles) | group | — | `10306b6a-6de5-4fcf-acb4-aad5e37ece3d___FAM_B.Msls 3918.JPG` |
| train | Grape___Esca_(Black_Measles) | group | _flipLR | `10306b6a-6de5-4fcf-acb4-aad5e37ece3d___FAM_B.Msls 3918_flipLR.JPG` |
| train | Grape___Esca_(Black_Measles) | group | — | `513edc8d-76f5-4604-bb7b-90195a600b60___FAM_B.Msls 4016.JPG` |
| train | Grape___Esca_(Black_Measles) | group | _flipLR | `513edc8d-76f5-4604-bb7b-90195a600b60___FAM_B.Msls 4016_flipLR.JPG` |
| test | Grape___Esca_(Black_Measles) | clean-test | — | `516e48d7-37c1-4c21-8ac8-d0a1659bbdde___FAM_B.Msls 0979.JPG` |
| train | Grape___Esca_(Black_Measles) | fill | — | `4b1f2354-fd30-4a68-84df-c6efc783b65a___FAM_B.Msls 1158.JPG` |
| test | Grape___Esca_(Black_Measles) | fill | — | `eb6f0cb7-781d-4b62-865f-ed7251ccbdf7___FAM_B.Msls 4009.JPG` |
| train | Grape___Esca_(Black_Measles) | fill | — | `23fd32ac-c77d-4f4f-8727-c66cfcce9717___FAM_B.Msls 3842.JPG` |
| test | Grape___Esca_(Black_Measles) | fill | — | `72a24fb2-e95f-4355-8122-896da4e3e515___FAM_B.Msls 4417.JPG` |
| train | Grape___Esca_(Black_Measles) | fill | — | `df33332e-be7b-4217-b6d3-bc986ece79c9___FAM_B.Msls 4441.JPG` |
| test | Grape___Esca_(Black_Measles) | fill | — | `caca5eaa-96b1-4fd5-8891-14e3edd34e10___FAM_B.Msls 1735.JPG` |
| train | Grape___Esca_(Black_Measles) | fill | — | `7dff376e-edf6-44c4-9074-eda8aa5f5deb___FAM_B.Msls 0951.JPG` |
| test | Grape___Esca_(Black_Measles) | fill | — | `8fe9bc15-1894-49a0-aa41-920920c5bd32___FAM_B.Msls 4251.JPG` |
| train | Grape___Esca_(Black_Measles) | fill | — | `a1c1aa92-4836-4b07-ade1-9da06a7b6f71___FAM_B.Msls 1015.JPG` |
| test | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | leak | — | `200b0ce7-899b-4cfe-b36b-1202ad34bc50___FAM_L.Blight 1617.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | leak | _flipLR | `200b0ce7-899b-4cfe-b36b-1202ad34bc50___FAM_L.Blight 1617_flipLR.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | leak | — | `2c135c30-428c-4ca3-a86f-5aa1b8e32aa9___FAM_L.Blight 1573.JPG` |
| test | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | leak | _flipLR | `2c135c30-428c-4ca3-a86f-5aa1b8e32aa9___FAM_L.Blight 1573_flipLR.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | group | — | `9f1994ba-44c4-4b33-9f3d-52da339602d5___FAM_L.Blight 3733.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | group | _flipLR | `9f1994ba-44c4-4b33-9f3d-52da339602d5___FAM_L.Blight 3733_flipLR.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | group | — | `796fa69b-4939-405b-9a83-21886ab3d087___FAM_L.Blight 3646.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | group | _flipLR | `796fa69b-4939-405b-9a83-21886ab3d087___FAM_L.Blight 3646_flipLR.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | group | — | `b517ea7d-7cb4-410c-899c-667270b45246___FAM_L.Blight 1568.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | group | _flipLR | `b517ea7d-7cb4-410c-899c-667270b45246___FAM_L.Blight 1568_flipLR.JPG` |
| test | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | clean-test | — | `347efc0d-b067-4c8a-8ee9-df88b70bbe2d___FAM_L.Blight 4628.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | fill | — | `d262bdfa-45b5-4166-a1e2-38c4cf79534d___FAM_L.Blight 4709.JPG` |
| test | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | fill | — | `7a19ccfa-2eed-4e6e-bb02-28064245874c___FAM_L.Blight 3661.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | fill | — | `ed5c8c75-0e44-4f65-b25c-05fb69b16655___FAM_L.Blight 1592.JPG` |
| test | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | fill | — | `d5d8d384-b7f6-4bb1-be43-2ddd0fd30dae___FAM_L.Blight 3701.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | fill | — | `2b3130c1-4245-4b5e-95ed-4d9b652cc86d___FAM_L.Blight 1727.JPG` |
| test | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | fill | — | `93467d1b-4121-4b65-804f-299fbe993eac___FAM_L.Blight 4708.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | fill | — | `01445290-a36f-4435-9b7d-bd1a9d1ff79b___FAM_L.Blight 4626.JPG` |
| test | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | fill | — | `7a8b7684-7062-4911-b639-8b0ab5b3ca2e___FAM_L.Blight 1710.JPG` |
| train | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | fill | — | `abef2d8b-9537-4902-a6eb-b7509bbfeb1a___FAM_L.Blight 4553.JPG` |
| test | Grape___healthy | leak | — | `c3b72cd8-977b-4a6e-a833-7bcc87873882___Mt.N.V_HL 9068.JPG` |
| train | Grape___healthy | leak | _180deg | `c3b72cd8-977b-4a6e-a833-7bcc87873882___Mt.N.V_HL 9068_180deg.JPG` |
| train | Grape___healthy | leak | — | `69a50f71-f8ee-4280-ac3e-c1d68e950c2e___Mt.N.V_HL 6165.JPG` |
| test | Grape___healthy | leak | _180deg | `69a50f71-f8ee-4280-ac3e-c1d68e950c2e___Mt.N.V_HL 6165_180deg.JPG` |
| train | Grape___healthy | group | — | `698a1cd5-2ad0-4a76-bb73-65918083df56___Mt.N.V_HL 6160.JPG` |
| train | Grape___healthy | group | _180deg | `698a1cd5-2ad0-4a76-bb73-65918083df56___Mt.N.V_HL 6160_180deg.JPG` |
| train | Grape___healthy | group | — | `4bd1e0f6-b69e-4936-83a2-235e6f8d7479___Mt.N.V_HL 6044.JPG` |
| train | Grape___healthy | group | _180deg | `4bd1e0f6-b69e-4936-83a2-235e6f8d7479___Mt.N.V_HL 6044_180deg.JPG` |
| train | Grape___healthy | group | — | `b6c49725-ccb9-46a9-abca-c99f51e13d0a___Mt.N.V_HL 6026.JPG` |
| train | Grape___healthy | group | _180deg | `b6c49725-ccb9-46a9-abca-c99f51e13d0a___Mt.N.V_HL 6026_180deg.JPG` |
| train | Grape___healthy | fill | — | `ef0cd637-5b34-489e-affc-72cce3dbc901___Mt.N.V_HL 6138.JPG` |
| test | Grape___healthy | fill | — | `aee361f6-d591-4c9f-9c66-a16c22523f8a___Mt.N.V_HL 9107.JPG` |
| train | Grape___healthy | fill | — | `a23b47a4-048d-4992-bc61-f3adec202543___Mt.N.V_HL 6113.JPG` |
| test | Grape___healthy | fill | — | `d966ee8e-abdc-4841-94a5-ae1bed9da16c___Mt.N.V_HL 9094.JPG` |
| train | Grape___healthy | fill | — | `515937e8-f329-406e-8d77-c9ddba3e399b___Mt.N.V_HL 9098.JPG` |
| test | Grape___healthy | fill | — | `0eb2593a-5e67-4f13-91ea-105831323d54___Mt.N.V_HL 6117.JPG` |
| train | Grape___healthy | fill | — | `1c3b3f51-d8c9-4bc5-899c-dd50e107f074___Mt.N.V_HL 6027.JPG` |
| test | Grape___healthy | fill | — | `a6868cab-63b1-4c6e-978a-136cef8181ab___Mt.N.V_HL 6109.JPG` |
| train | Grape___healthy | fill | — | `2211cfa2-311c-41f0-aba5-9bdef8037f20___Mt.N.V_HL 6051.JPG` |
| test | Grape___healthy | fill | — | `bfeeb99b-946a-4edb-b59e-445d682d891d___Mt.N.V_HL 6061.JPG` |
