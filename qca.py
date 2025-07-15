import pandas as pd
import numpy as np
from pprint import pprint as pp
import qm_tools_aw
from qm_tools_aw import qca
import qcelemental as qcel
from qcportal import PortalClient
from pprint import pprint as pp
from qcportal.singlepoint import (
    SinglepointDataset,
    SinglepointDatasetEntry,
    QCSpecification,
)

client = qm_tools_aw.qca.establish_client("http://localhost:7777", verify=False)
# client = PortalClient("http://localhost:7777", verify=False)
print(client)


def setup_entries():
    df = pd.read_pickle("./plots/ddft_study.pkl")
    df["Geometry"] = df["Geometry"].apply(lambda x: np.array(x))
    df["monAs"] = df["monAs"].apply(lambda x: np.array(x))
    df["monBs"] = df["monBs"].apply(lambda x: np.array(x))
    # pp(df.columns.tolist())
    # return

    def create_qcel_mol(r):
        ma, mb = r["monAs"], r["monBs"]
        g1, g2 = r["Geometry"][ma], r["Geometry"][mb]
        m1 = f"{r['charges'][1][0]} {r['charges'][1][1]}\n"
        m1 += qm_tools_aw.tools.print_cartesians_pos_carts(
            g1[:, 0], g1[:, 1:], only_results=True
        )
        m2 = f"{r['charges'][2][0]} {r['charges'][2][1]}\n"
        m2 += qm_tools_aw.tools.print_cartesians_pos_carts(
            g2[:, 0], g2[:, 1:], only_results=True
        )
        mol = qcel.models.Molecule.from_data(
            m1 + "--\n" + m2,
            extras={
                "id": "id",
                "system_id": str(r["system_id"]),
                "DB": str(r["DB"]),
                "R": str(r["R"]),
                "Benchmark": float(r["Benchmark"]),
                # PBE0
                "SAPT_DFT_pbe0_adz_elst": float(r["SAPT_DFT_pbe0_adz_elst"]),
                "SAPT_DFT_pbe0_adz_exch": float(r["SAPT_DFT_pbe0_adz_exch"]),
                "SAPT_DFT_pbe0_adz_indu": float(r["SAPT_DFT_pbe0_adz_indu"]),
                "SAPT_DFT_pbe0_adz_disp": float(r["SAPT_DFT_pbe0_adz_disp"]),
                "pbe0_adz_grac_A": float(r["pbe0_adz_grac_A"]),
                "pbe0_adz_grac_B": float(r["pbe0_adz_grac_B"]),
                "SAPT_DFT_pbe0_adz_dHF": float(r["SAPT_DFT_pbe0_adz_dHF"]),
                "SAPT_DFT_pbe0_adz_dDFT": float(r["SAPT_DFT_pbe0_adz_dDFT"]),
                "SAPT_DFT_pbe0_adz_D4_IE": float(r["SAPT_DFT_pbe0_adz_D4_IE"]),
                "SAPT_DFT_pbe0_adz_DFT_IE": float(r["SAPT_DFT_pbe0_adz_DFT_IE"]),
                "pbe0_atz_grac_A": float(r["pbe0_atz_grac_A"]),
                "pbe0_atz_grac_B": float(r["pbe0_atz_grac_B"]),
                # B3LYP
                "SAPT_DFT_pbe0_atz_elst": float(r["SAPT_DFT_pbe0_atz_elst"]),
                "SAPT_DFT_pbe0_atz_exch": float(r["SAPT_DFT_pbe0_atz_exch"]),
                "SAPT_DFT_pbe0_atz_indu": float(r["SAPT_DFT_pbe0_atz_indu"]),
                "SAPT_DFT_pbe0_atz_disp": float(r["SAPT_DFT_pbe0_atz_disp"]),
                "SAPT_DFT_pbe0_atz_dHF": float(r["SAPT_DFT_pbe0_atz_dHF"]),
                "SAPT_DFT_pbe0_atz_dDFT": float(r["SAPT_DFT_pbe0_atz_dDFT"]),
                "SAPT_DFT_pbe0_atz_D4_IE": float(r["SAPT_DFT_pbe0_atz_D4_IE"]),
                "SAPT_DFT_pbe0_atz_DFT_IE": float(r["SAPT_DFT_pbe0_atz_DFT_IE"]),
                "b3lyp_adz_grac_A": float(r["b3lyp_adz_grac_A"]),
                "b3lyp_adz_grac_B": float(r["b3lyp_adz_grac_B"]),
                "SAPT_DFT_b3lyp_adz_elst": float(r["SAPT_DFT_b3lyp_adz_elst"]),
                "SAPT_DFT_b3lyp_adz_exch": float(r["SAPT_DFT_b3lyp_adz_exch"]),
                "SAPT_DFT_b3lyp_adz_indu": float(r["SAPT_DFT_b3lyp_adz_indu"]),
                "SAPT_DFT_b3lyp_adz_disp": float(r["SAPT_DFT_b3lyp_adz_disp"]),
                "SAPT_DFT_b3lyp_adz_dHF": float(r["SAPT_DFT_b3lyp_adz_dHF"]),
                "SAPT_DFT_b3lyp_adz_dDFT": float(r["SAPT_DFT_b3lyp_adz_dDFT"]),
                "SAPT_DFT_b3lyp_adz_D4_IE": float(r["SAPT_DFT_b3lyp_adz_D4_IE"]),
                "SAPT_DFT_b3lyp_adz_DFT_IE": float(r["SAPT_DFT_b3lyp_adz_DFT_IE"]),
                "SAPT_DFT_b3lyp_atz_elst": float(r["SAPT_DFT_b3lyp_atz_elst"]),
                "SAPT_DFT_b3lyp_atz_exch": float(r["SAPT_DFT_b3lyp_atz_exch"]),
                "SAPT_DFT_b3lyp_atz_indu": float(r["SAPT_DFT_b3lyp_atz_indu"]),
                "SAPT_DFT_b3lyp_atz_disp": float(r["SAPT_DFT_b3lyp_atz_disp"]),
                "SAPT_DFT_b3lyp_atz_dHF": float(r["SAPT_DFT_b3lyp_atz_dHF"]),
                "SAPT_DFT_b3lyp_atz_dDFT": float(r["SAPT_DFT_b3lyp_atz_dDFT"]),
                "SAPT_DFT_b3lyp_atz_D4_IE": float(r["SAPT_DFT_b3lyp_atz_D4_IE"]),
                "SAPT_DFT_b3lyp_atz_DFT_IE": float(r["SAPT_DFT_b3lyp_atz_DFT_IE"]),
            },
        )
        return mol

    df["qcel_mol"] = df.apply(lambda r: create_qcel_mol(r), axis=1)
    print(df[["system_id", "qcel_mol"]])
    print(df["qcel_mol"].iloc[0])
    geoms = []
    for i, row in df.iterrows():
        geoms.append([row["system_id"], row["qcel_mol"]])
    ds = qca.init_singlepoint_dataset(
        client,
        ds_name="ddft_study",
        geoms=geoms,
    )
    return ds


def main():
    ds = client.get_dataset("singlepoint", "ddft_study")
    # client.delete_dataset(ds.id, delete_records=True)
    # setup_entries()
    # return
    # ds = qca.init_singlepoint_dataset(
    #     client,
    #     ds_name="ddft_study",
    # )
    print(ds)
    functional = "pbe0"
    functional = "b3lyp"
    basis = "aug-cc-pvdz"
    # basis = "aug-cc-pvtz"
    # return
    if False:
        qca.create_singlepoint_dataset_specification(
            ds,
            program="psi4",
            driver="energy",
            method="sapt(dft)",
            basis=basis,
            keywords={
                "maxiter": 250,
                "E_CONVERGENCE": 8,
                "D_CONVERGENCE": 8,
                "freeze_core": "True",
                "guess": "sad",
                "scf_type": "df",
                "SAPT_DFT_FUNCTIONAL": functional,
                "SAPT_DFT_DO_DISP": True,
                "SAPT_DFT_DO_DDFT": True,
                "SAPT_DFT_D4_IE": True,
                "SAPT_DFT_GRAC_COMPUTE": "ITERATIVE",
            },
            protocols={"stdout": True},
            specification_name=f"psi4/sapt({functional})/aug-cc-pvdz",
            compute_tag="hive",
        )
    print(ds.status())
    problem_ids = ids = [53753, 53829, 53814, 53833, 53788, 53813, 53808, 53895, 53736, 53901, 53838, 53826, 53846, 53758, 53861, 53785, 53748, 53906, 53761, 53799, 53795, 53858, 53832, 53863, 53773, 53752, 53851, 53778, 53802, 53771, 53891, 53959, 53938, 53943, 53969, 53927, 53946, 53972, 53953, 53923, 53945, 53956, 53955, 53948, 54106, 54147, 54071, 54108, 54074, 54066, 54131, 54042, 54095, 54122, 53985, 54050, 54130, 54160, 53994, 54091, 54021, 54142, 54072, 54105, 54023, 54057, 54047, 54100, 54088, 54219, 54202, 54214, 54198, 54636, 54512, 54625, 54524, 54588, 54486, 54284, 54418, 54325, 54415, 54273, 54350, 54312, 54414, 54369, 54384, 54383, 54344, 54360, 54404, 54269, 54397, 54366, 54268, 54405, 54379, 54413, 54376, 54274, 54336, 54345, 54307, 54337, 54235, 54461, 54443, 54450, 54431, 54602, 54557, 54494, 54503, 54490, 54551, 54475, 54532, 54485, 54613, 54654, 54545, 54511, 54498, 54640, 54661, 54600, 54519, 54487, 54481, 54630, 54691, 54696, 54842, 54896, 54725, 54869, 55009, 54993, 55057, 54819, 54891, 54801, 54907, 54836, 54912, 54744, 54737, 54909, 54732, 54733, 54813, 54878, 54787, 54904, 54847, 54800, 54759, 54851, 54883, 54779, 54884, 54771, 54900, 54829, 54755, 54835, 54930, 54925, 54950, 55402, 55364, 55226, 55308, 55286, 55277, 55114, 55087, 54975, 55117, 55089, 55161, 55171, 55062, 55128, 55109, 55146, 54987, 55172, 54990, 55143, 55102, 55116, 55007, 55069, 55101, 55082, 54989, 55015, 55133, 55066, 55112, 55022, 55076, 55051, 55104, 55080, 55003, 55183, 55187, 55184, 55177, 55200, 55179, 55204, 55198, 55208, 55182, 55178, 55173, 55567, 55646, 55653, 55554, 55519, 55606, 55497, 55544, 55265, 55247, 55403, 55246, 55290, 55329, 55254, 55315, 55281, 55312, 55303, 55422, 55251, 55360, 55299, 55393, 55377, 55238, 55302, 55239, 55407, 55227, 55223, 55349, 55234, 55321, 55273, 55372, 55461, 55463, 55470, 55577, 55607, 55652, 55644, 55648, 55579, 55643, 55479, 55588, 55604, 55661, 55575, 55721, 55687, 55715, 55712, 55673, 55713, 55714, 55696, 55692, 55919, 55911, 55818, 55734, 55902, 55788, 55878, 55811, 55754, 55906, 55809, 55828, 55918, 55833, 55875, 55794, 55723, 55848, 55896, 55781, 55738, 55913, 55826, 55893, 55725, 55758, 55771, 55970, 55934, 55924, 55942, 55961, 55953, 55957, 55959, 56277, 56307, 56366, 56381, 56276, 56164, 55979, 56010, 56023, 56001, 56089, 55986, 56152, 56169, 56132, 56055, 55974, 56060, 56000, 56112, 56033, 56083, 56108, 56011, 56019, 56121, 55996, 56160, 55984, 56155, 56063, 56168, 56222, 56216, 56221, 56547, 56541, 56563, 56308, 56267, 56257, 56310, 56360, 56295, 56404, 56401, 56296, 56242, 56315, 56239, 56394, 56346, 56238, 56234, 56326, 56334, 56447, 56465, 56434, 56471, 56445, 56457, 56448, 56464, 56462, 56425, 56437, 56583, 56529, 56559, 56521, 56527, 56522, 56565, 56639, 56519, 56625, 56599, 56476, 56480, 56553, 56498, 56536, 56633, 56474, 56612, 56550, 56478, 56695, 56718, 56710, 56681, 56715, 56676, 56678, 56706, 56823, 56736, 56733, 56759, 56724, 56867, 56921, 56732, 57118, 57019, 57133, 57058, 56981, 57075, 56883, 56803, 56740, 56840, 56900, 56849, 56954, 56955, 56948, 56945, 56940, 56944, 57241, 57120, 56998, 57168, 57170, 57010, 57051, 57100, 56975, 57140, 57063, 57021, 57176, 57204, 57193, 57206, 57184, 57215, 57175, 57213, 57553, 57569, 57535, 57613, 57607, 57501, 57574, 57624, 57235, 57394, 57412, 57324, 57398, 57290, 57244, 57393, 57377, 57276, 57264, 57268, 57258, 57288, 57279, 57365, 57295, 57335, 57363, 57402, 57413, 57416, 57374, 57444, 57467, 57448, 57453, 57451, 57456, 57430, 57551, 57564, 57647, 57541, 57477, 57650, 57605, 57603, 57591, 57506, 57620, 57599, 57584, 57565, 57549, 57701, 57717, 57678, 57696, 57698, 57684, 57807, 57731, 57884, 57990, 57996, 58057, 58144, 58016, 58048, 58055, 58049, 58019, 57998, 58010, 58161, 58076, 57742, 57829, 57898, 57859, 57777, 57769, 57915, 57735, 57766, 57920, 57870, 57911, 57873, 57872, 57825, 57877, 57906, 57963, 57989, 58045, 58147, 58170, 58037, 58138, 58042, 58143, 58096, 58063, 58047, 58044, 57985, 58156, 58167, 58078, 58176, 58221, 58237, 58276, 58272, 58229, 58225, 58271]
    # Get id of records that failed
    # print(ds.detailed_status())
    # recs = client.get_records(record_ids=ids, include=['entry_name'])
    # client.reset_records(ids)
    # return
    # get errors

    conv_str = "Convergence error, trying next GRAC iteration..."
    convergence_issues_1 = 0
    convergence_issues_2 = 0
    convergence_issues_3 = 0
    # for n, entry in enumerate(ds.detailed_status()):
        # name, lot, status = entry
    ids = []
    # for n, (e, s, r) in enumerate(ds.iterate_records(specification_names=["psi4/sapt(pbe0)/aug-cc-pvdz"])):
    for n, (e, s, r) in enumerate(ds.iterate_records(specification_names=["psi4/sapt(b3lyp)/aug-cc-pvdz"])):
        if str(r.status) == "RecordStatusEnum.error":
            # print(name, lot)
            # r = ds.get_record(name, lot)
            # print(r.stdout)
            # print(r.stderr)
            # print(r.error)
            # print()
            ids.append(r.id)
            pass
        else:
            # print(r.stdout)
            cnt = r.stdout.count(conv_str)
            if cnt > 0:
                print(r)
                print("Convergence errors:", cnt)
                print()
                ids.append(r.id)
                if cnt == 1:
                    convergence_issues_1 += 1
                elif cnt == 2:
                    convergence_issues_2 += 1
                elif cnt == 3:
                    convergence_issues_3 += 1
        if n % 100 == 0:
            print(f"{n} records processed.")
        # if n > 10:
        #     break
    print("Convergence struggles:", convergence_issues_1 + convergence_issues_2 + convergence_issues_3)
    print("Convergence issues with 1 iteration:", convergence_issues_1)
    print("Convergence issues with 2 iterations:", convergence_issues_2)
    print("Convergence issues with 3 iterations:", convergence_issues_3)
    print(f"{ids = }")
    return


if __name__ == "__main__":
    main()
