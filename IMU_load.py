import numpy as np
import pandas as pd
import struct
import os

def readIMAR_Header(*args): 
    """
    [fields] = readIMAR(ifile)
    Reads the header of the output files from the iMAR unit (.Dat files).
    This function is based upon the script made by Tim Jensen (DTU) 24/05-2018

    -------------------------------------------------------------------------
    Input:
    ifile       Path and name of the .dat file

    Optional input arguments:
    hfunc       Input argument related to optional functions ("echo")
    harg        Argument related to the optional function ("on"/"off")

    Output:
    alldata     Matlab cell containing an array of data for each column
    fields      Matlab cell containing Matlab structures with information
                regarding the data in each column
    header      Matlab cell containing all the original header information
    sync        Matlab cell with sync byte information of iMAR binary data
                format

     NOTES:     If (FLAG!!!) appears in the final header, then the reading
                algorithm has encounted a character it was not able to 
                decode. 
    -------------------------------------------------------------------------
    Author: Christian Solgaard (DTU Master student)
    """
    
    # Check inputs
    if len(args) != 3:
        raise ValueError("Expected 3 arguments, but got {}".format(len(args)))
    ifile, hfunc, harg = args

    # Check if file exists 
    if not os.path.isfile(ifile):
        raise Exception(ifile + ' does not exist!')

    # Set default values 
    h_echo = True

    # Change default values if needed
    if len(args) > 1: 
        if hfunc.lower() == "echo": 
            if harg.lower() == "off":
                h_echo = False
    
    # Echo Procedure 
    if h_echo:
        print("Reading header of file:", ifile) 

    # -----------------------------Open the File ---------------------------------
    with open(ifile, "rb") as fid: 

        ## Read header information (ascii)
        # Initialise
        header = {"column":[], "pre_column":[], "post_column":[], "n_column":[]}  # dic for header lines
        n_pre_column = 0  # Pre-column header counter
        n_column = 0  # Column counter
        n_post_column = 0  # Post-columns header counter
        fields = {"group":[], "name":[], "unit":[], "scale":[], "bias":[], 
                "bytesize":[], "format":[], "format_str":"<", "bytesPerRow":[]}  # dic for column information
        bytesPerRow = 0  # Counter for bytes per row
        lcol = False  # Logical indicator for when column info is reached


        # Read header line by line 
        while True: 
            # Read next line 
            tline=fid.readline()
            try:
                # Try to encode the line with UTF-8 encoding
                tline = tline.decode()
            except UnicodeDecodeError as e:
                # If encoding fails, insert the flag string in the position of the error
                tline = tline.decode(errors="ignore")
                tline = tline[:e.start] + "(FLAG!!!)" + tline[e.end-1:]
            
            # Check if string is empty
            if not tline or tline == "\r\n" or tline == "\n":
                break
            
            # Check if string contains column information
            if tline[0:8].lower() == "# column": 

                # Increase counter 
                n_column += 1

                # Store header contents
                header["column"].append(tline[12::])

                # Divide columns seperated by ","
                entries = tline.split(',')

                # Extract field name
                fields["group"].append(entries[0][12:].strip())

                # Extract header info into Matlab cell array
                if entries[0][12:].strip().lower() == "marker":

                    # Change logical indicator
                    lcol = 1

                    # Store line information in python dictionary structure
                    fields["name"].append(fields["group"][-1])
                    str_ = entries[1].strip()
                    fields["bytesize"].append(int(str_[1:2]))
                    fields["format"].append(str_[str_.find('byte') + 5 : str_.find(')')])

                    if fields["format"][-1].lower() == "int": 
                        str_in = "i"
                    elif fields["format"][-1].lower() == "float": 
                        str_in = "f"
                    else: 
                        str_in = "d"
                
                    fields["format_str"] += str_in
                    
                    # fields["class"].append(f"uint{fields['bytesize'][-1]}")

                    # Compute number of bytes per row
                    bytesPerRow = bytesPerRow + fields["bytesize"][-1]

                else: 
                    # Store line information in python dictionary structure
                    fields["name"].append(entries[1].strip())
                    fields["unit"].append((entries[2][entries[2].find('[')+1 : entries[2].find(']')]).strip())
                    fields["scale"].append(float(entries[3][entries[3].find('=')+1 : entries[3].find('*')]))
                    fields["bias"].append(float(entries[3][entries[3].find('-')+1 ::]))
                    str_ = entries[4].strip()
                    fields["bytesize"].append(int(str_[1:2]))
                    fields["format"].append(str_[str_.find('byte') + 5 : str_.find(')')])

                    # Determine the precision of the data
                    
                    if fields["format"][-1].lower() == "int": 
                        str_in = "i"
                    elif fields["format"][-1].lower() == "float": 
                        str_in = "f"
                    else: 
                        str_in = "d"
                
                    fields["format_str"] += str_in

                    # Compute number of bytes per row 
                    bytesPerRow = bytesPerRow + fields["bytesize"][-1]
            
            else: 
                if lcol: 
                    # Store header contents 
                    header["post_column"].append(tline)

                else: 
                    # Store header contents 
                    header["pre_column"].append(tline)
            
                # Extract sync information
                if tline[0:11].lower() == "# sync_byte":
                    # Divide columns seperated by ","
                    entries = tline.split(',')
                    sync = []
                    sync.append(int((entries[0][entries[0].find('x')-1::]).strip(),16))
                    sync.append(float(entries[1][entries[1].find('=')+1::]))
                    sync.append(float(entries[2][entries[2].find('=')+1::]))
                    sync.append(int((entries[3][entries[3].find('x')-1::]).strip(),16))
        header["n_column"] = n_column
        fields["bytesPerRow"] = bytesPerRow
    return fields, header, sync

def readIMAR_Data(*args): 
    """
    [alldata,fields] = readIMAR(ifile)
    Reads output files from the iMAR unit (.Dat files).
    -------------------------------------------------------------------------
    Input:
    ifile       Path and name of the .dat file
    sync        Matlab cell with sync byte information of iMAR binary data
                format

    Optional input arguments:
    hfunc       Input argument related to optional functions
    harg        Argument related to the optional function

    Output:
    alldata     Matlab cell containing an array of data for each column
    -------------------------------------------------------------------------
    Author: Christian Solgaard (DTU Master student)
    """
    
    # Check inputs
    if len(args) != 5:
        raise ValueError("Expected 5 arguments, but got {}".format(len(args)))
    ifile, hfunc, harg, sync, fields = args

    # Check if file exists 
    if not os.path.isfile(ifile):
        raise Exception(ifile + ' does not exist!')
    
    # Set path to output file, in parent folder. 
    input_path = ifile
    
    # Extract the input filename and path
    input_filename = os.path.basename(input_path)
    input_dir = os.path.dirname(input_path)

    # Extract the number from the input filename
    number = input_filename.split('_')[0]

    # Construct the output filename
    output_filename = f"{number}_imar.pkl"
    output_path = os.path.join(input_dir, output_filename)

    # Set default values 
    h_echo = True

    # Change default values if needed
    if len(args) > 1: 
        if hfunc.lower() == "echo": 
            if harg.lower() == "off":
                h_echo = False
    
    # Echo Procedure 
    if h_echo:
        print("Reading data of file:", ifile) 

    # -----------------------------Open the File ---------------------------------
    ## Read data columns (binary)
    row_count = 0
    with open(ifile, 'rb') as fid:
        # Read the header line and discard it
        header_terminate = sync[-1].to_bytes(1, "little")
        while True:
            byte = fid.read(1)
            if byte == header_terminate:
                break
        # Read each row of binary data
        data = []

        while True:
            row_count += 1
            if row_count % sync[2] == 1 and row_count > sync[2]:
                fid.read(int(sync[1]))
            # Read a row of binary data
            line = fid.read(fields["bytesPerRow"])
            
            if len(line) < fields["bytesPerRow"]:
                break
            # Unpack the data using the format string
            row = struct.unpack(fields["format_str"], line)
            
            # Append the row to the data list
            data.append(row)
        data = np.array(data)
    
    # Create a sample DataFrame
    df = pd.DataFrame({'time': data[:,1], 'bacc1': data[:,2], 'bacc2': data[:,3], 'bacc3': data[:,4]})

    # data_dict = {fields["name"][i]: data[:, i] for i in range(len(fields["name"]))}
    # df = pd.DataFrame(data_dict)

    df.to_pickle(output_path)
    print("Decoded file created at :", output_path)
    return 

def readIMAR(*args): 
    """
    Procedure to read and convert the hexadecimal formatted data from the 
    IMAR IMU output file. The converted file, is located at the 
    origin path of the .dat file. 

    The readIMAR function uses two different subrutines: readIMAR_Header()
    which reads and saves important information from the header of the .dat
    file, and readIMAR_Data() which loads the data based upon the information
    transfered form the readIMAR_Header() function. 
    -------------------------------------------------------------------------
    Input:
    ifile       Path and name of the .dat file

    Optional input arguments:
    hfunc       Input argument related to optional functions ("echo")
    harg        Argument related to the optional function ("on"/"off")

    Output:
    data        dataframe with columns: "time", "bacc1", "bacc2", "bacc3".
    .pkl file:  Datafile in the pkl file format, special fileformat for python
                Located at the original datafile location. 

    """
    
    # Check inputs
    if len(args) != 3:
        raise ValueError("Expected 3 arguments, but got {}".format(len(args)))
    ifile, hfunc, harg = args

    # Check if file exists 
    if not os.path.isfile(ifile):
        raise Exception(ifile + ' does not exist!')
    
    # Set path to output file, in parent folder. 
    input_path = ifile
    
    # Extract the input filename and path
    input_filename = os.path.basename(input_path)
    input_dir = os.path.dirname(input_path)

    # Extract the number from the input filename
    number = input_filename.split('_')[0]

    # Construct the output filename
    output_filename = f"{number}_imar.pkl"
    output_path = os.path.join(input_dir, output_filename)

    # Check if outputfile already exist. 
    if os.path.isfile(output_path):
        while True:
            overwrite = input("Output file already exists. Do you want to overwrite it? (y/n) ").lower()
            if overwrite == "y":
                break
            elif overwrite == "n":
                data = pd.read_pickle(output_path)
                print("Read data from file located at: ", output_path)
                return data
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    fields, header, sync = readIMAR_Header(ifile, hfunc, harg) # Read Header information
    readIMAR_Data(ifile, hfunc, harg, sync, fields) # Read and save binary data format to pickle file.
    data = pd.read_pickle(output_path) # Load the pickle data file
    print("File created and read from location: ", output_path) 
    return data
    


def load_gnss(ifile):
    """
    Loading function for the ppp gnss file, in .txt format 

    ---------------------------------------------------------------
    Input: 
    ifile:              Path and name of file ___.txt
    
    Output: 
    gnss:                Pandas dataframe format containing the columns: 
                        ["lat", "lon", "h", "time"]
    
    ----------------------------------------------------------------
    Author: Christian Solgaard (DTU - Master student) 17/02-2023
    """ 
    # Check if file exists 
    if not os.path.isfile(ifile):
        raise Exception(ifile + ' does not exist!')

    # Echo
    print("Reading GNSS datafile: ", ifile)

    ## Loading GNSS File
    with open(ifile) as fid:
        while True: 
            tline = fid.readline()

            if tline[2:8].lower() == "seqnum": 
                columns = tline.split()
                break

    ppp = pd.read_csv(ifile.as_posix(), skiprows=48, delim_whitespace=True, 
                      header=None, names= columns)
    gnss = ppp[['Latitude', 'Longitude', 'H-Ell', 'CorrTime']].copy()
    gnss.columns = ["lat", "lon", "h", "time"]

    return gnss


def load_nav(ifile):
    """
    Loading function for the navigation file, in .txt format 

    ---------------------------------------------------------------
    Input: 
    ifile:              Path and name of file ___.txt
    
    Output: 
    nav:                Pandas dataframe format containing the columns: 
                    ["lat", "lon", "h", "vn", "ve", "vd", "roll", "pitch", "yaw", "time"]
    header:             Header information. 
    ----------------------------------------------------------------
    Author: Christian Solgaard (DTU - Master student) 17/02-2023
    """ 
    # Check if file exists 
    if not os.path.isfile(ifile):
        raise Exception(ifile + ' does not exist!')

    # Echo
    print("Reading NAV datafile: ", ifile)


    ## Loading Navigation File
    with open(ifile) as fid:

        line_count = 0
        while True: 
            tline = fid.readline()
            line_count += 1

            if tline[2:8].lower() == "seqnum": 
                columns = tline.split()
                line_count += 1
                break

    air1 = pd.read_csv(ifile.as_posix(), skiprows=line_count, delim_whitespace=True, header=None,
                    names=columns)
    nav = air1[["Latitude", "Longitude", "H-Ell","VNorth", "VEast", "VUp", "Roll", 
                "Pitch", "Heading","CorrTime"]].copy()
    nav.columns = ["lat", "lon", "h", "vn", "ve", "vd", "roll", "pitch", "yaw", "time"]

    return nav


