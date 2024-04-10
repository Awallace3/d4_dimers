import src


def main():
    # df_name = "dfs/los_saptdft_atz.pkl"
    df_name = "./dfs/los_saptdft_atz_1.pkl"
    src.plotting.plotting_setup_dft_ddft(
        df_name,
        build_df=True,
        split_components=True,
    )
    return


if __name__ == "__main__":
    main()
