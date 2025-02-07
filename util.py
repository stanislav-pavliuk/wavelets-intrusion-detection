from curses import start_color
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import pywt
from torch import normal


def read_files(filenames: list):
    return pd.concat([pd.read_csv(f'./data/{filename}') for filename in filenames])


def read_all_files():
    dir = os.listdir('./data/')

    filenames = [filename for filename in dir if filename.endswith('.csv')]
    
    return read_files(filenames)


def dwt_multilevel_transform(data, wavelet='coif5'):
    """
    Perform a multilevel discrete wavelet transform on the data

    Args:
        data (pd.DataFrame): Initial dataframe
        wavelet (str, optional): Defaults to 'db28'.
    
    Returns:
        pd.DataFrame: transformed dataframe
    """

    # swt_coefficients = {}
    
    # for column in data.columns:
    #     cA, cB = pywt.swt(data[column], wavelet=wavelet, level=1)[0]

    #     swt_coefficients[column + '_approx'] = cA
    #     swt_coefficients[column + '_detail'] = cB
        
    # swt_df = pd.DataFrame(swt_coefficients)
    
    # return swt_df[0]['aa']

    return pywt.swt(data, wavelet=wavelet, level=1)[0]

def remove_background(normal_internal_resampled, rank = 1):
    """Remove the most significant rank singular values from the data

    Args:
        normal_internal_resampled (pd.DataFrame): Initial dataframe
        rank (int, optional):  Defaults to 2.

    Returns:
        pd.DataFrame: reduced dataframe
    """
    # scaler = StandardScaler()
    # reduced_df = scaler.fit_transform(normal_internal_resampled)

    # return reduced_df

    reduced_df = normal_internal_resampled

    # return reduced_df

    U, s, Vt = np.linalg.svd(reduced_df, full_matrices=False)

    rank_k = np.dot(U[:, :rank], np.dot(np.diag(s[:rank]), Vt[:rank, :]))

    # reduced_df = reduced_df - rank_k

    return rank_k



# not sure if this is the best way to handle the port values
def categorize_port(port):
    if port in [80, 443, 444, 8080, 8443]:  # Common web ports
        return 'web'
    elif port in [22, 23, 21]:  # SSH, Telnet, FTP
        return 'remote_access'
    elif port in [53]:  # DNS
        return 'dns'
    elif port in range(10, 1024):  # Well-known ports
        return 'well_known'
    elif port in range(1024, 49151):  # Registered ports
        return 'registered'
    elif port >= 49152:  # Dynamic/private ports
        return 'dynamic'
    else:
        return 'unknown'  # Other/unknown ports
    
def resample(df, freq='1s'):
    """
    Resample the dataframe to a given frequency
    
    Args:
        df (pd.DataFrame): Initial dataframe
        freq (str, optional): Defaults to '500ms'.
        
    Returns:
        pd.DataFrame: resampled dataframe
    """

    df = df.assign(tcp_count=lambda x: x['protocol'].eq("TCP").astype(int))
    df = df.assign(udp_count=lambda x: x['protocol'].eq("UDP").astype(int))
    
    df = df.assign(dst_port_category=lambda x: x['dst_port'].apply(categorize_port))
    df = df.assign(src_port_category=lambda x: x['src_port'].apply(categorize_port))

    df = df.assign(src_internal_ip_count=lambda x: x['src_ip'].str.contains('192.168').astype(int))
    df['dst_host_port_from_internal'] = np.where(
        df['src_ip'].str.startswith('192.168'),  # Condition
        df['dst_ip'].astype(str) + ':' + df['dst_port'].astype(str),  # If True
        np.nan  # If False
    )

    df = pd.get_dummies(df, columns=['dst_port_category', 'src_port_category'])

    return df.resample(freq).agg({
        # "udp_count": "sum",
        # "tcp_count": "sum",

        # "src_internal_ip_count": "sum",

        # "dst_ip": lambda x: len(np.unique(x)) / (len(x) + 1),
        # "dst_port": lambda x: 1 / (np.var(x) + 0.00000001),
        # "dst_host_port_from_internal": lambda x: len(np.unique(x.dropna())) / (len(x) + 1),

        # "dst_port": "nunique",
        # "dst_host_port": "nunique",

        
        # "fwd_payload_bytes_max": "max",
        # "fwd_payload_bytes_variance": "mean",
        

        # "duration": "median", 

        # "packets_count": "sum",
        # "fwd_packets_count": "sum",
        # "bwd_packets_count": "sum",

        # "payload_bytes_mean": "median",
        # "payload_bytes_std": "mean",
        
        # "total_header_bytes": "sum",
        # "total_payload_bytes": "sum",

        # "payload_bytes_std": lambda x: 1 / (np.mean(x) ** 2 + 0.00000001),

        "fwd_total_payload_bytes": lambda x: 1 / (np.std(x) + 0.00000001),
        
        # "active_mean": "mean",
        # "idle_mean": "mean",

        # "bytes_rate": "mean",
        # "packets_rate": "mean",
        # "down_up_rate": "mean",
        
        # "avg_bwd_bulk_rate": "mean",
        # "packets_IAT_mean": "mean",
        # "packet_IAT_std": "mean",
        # "packet_IAT_total": "mean",
        # "packet_IAT_std": "mean",


        # "ack_flag_counts": "sum",
        # "bwd_fin_flag_counts": "sum",
        "dst_port_category_web": "sum",
        # "bwd_rst_flag_counts": "sum",

        # "fwd_packets_IAT_total": "sum",

        # "payload_bytes_variance": "sum",

        # "fin_flag_counts": "sum",
        # "rst_flag_counts": "sum",

        # "src_port_category_dynamic": "sum",
        # "src_port_category_registered": "sum",
        # "src_port_category_remote_access": "sum",
        # "src_port_category_web": "sum",
        # "src_port_category_well_known": "sum",
        
        # "dst_port_category_web": "sum",
        # "dst_port_category_well_known": "sum",
        # "dst_port_category_dynamic": "sum",
        # "dst_port_category_registered": "sum",
        # "dst_port_category_remote_access": "sum",
        # "dst_port_category_well_known": "sum", 
    }).interpolate('time'), df['label'].resample(freq).agg(lambda x: 1 if x.eq("Benign").all() else -1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def preprocess(traffic_df, scaler, all=False):
    normal_internal = traffic_df
    if not all: 
        print("Filtering only internal traffic")
        normal_internal = traffic_df[traffic_df['dst_ip'].str.contains('192.168')]

    normal_internal_resampled, normal_internal_resampled_labeled = resample(normal_internal, freq='1s')

    normal_internal_resampled = pd.DataFrame(scaler(normal_internal_resampled), columns=normal_internal_resampled.columns)

    reduced_df = remove_background(normal_internal_resampled, rank=1)
    normal_internal_resampled = pd.DataFrame(reduced_df, columns=normal_internal_resampled.columns)

    if len(normal_internal_resampled) % 2 != 0:
        normal_internal_resampled = normal_internal_resampled[:-1]
        normal_internal_resampled_labeled = normal_internal_resampled_labeled[:-1]
        
    swt_coef = pywt.swtn(normal_internal_resampled, wavelet='db38', level=1, norm=True)[0]
    swt_df = pd.DataFrame(swt_coef['ad'][:, 1], columns=['ddos_loit'])

    
    # dos_approx, (dos_h, dos_v, dos_d) = pywt.swt2(normal_internal_resampled[['dst_port_category_web', 'fwd_total_payload_bytes']].values, 'coif17', level=1, start_level=0, norm=True, trim_approx=True)
    # port_approx, (port_h, port_v, port_d) = pywt.swt2(normal_internal_resampled[['dst_port', 'bwd_rst_flag_counts']].values, 'coif17', level=1, start_level=0, norm=True, trim_approx=True)
    # # botnet_approx, (botnet_h, botnet_v, botnet_d) = pywt.swt2(normal_internal_resampled[['dst_host_port_from_internal', 'payload_bytes_std']].values, 'coif17', level=1, start_level=0, norm=True, trim_approx=True)


    # swt = np.array((dos_d[:, 0])).T
    
    # swt_df = pd.DataFrame(swt, columns=["ddos_loit"])


    return swt_df, normal_internal_resampled_labeled