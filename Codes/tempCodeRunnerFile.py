import wrds

##############################################
#  This code teaches on the main functions   #
#  to connect to WRDS server using library   #
#       and a bunch of fun stuff             #
##############################################
conn = wrds.Connection()

# List all libraries
library_list = conn.list_libraries()
print("Available libraries:")
print(library_list)

# List datasets within a specific library
library_name = 'taqm_2021'  # Change to the desired library
table_list = conn.list_tables(library=library_name)
print(f"Datasets in library '{library_name}':")
print(table_list)

# Explore the structure of a specific table within the library
# Replace 'your_table_name' with the name of the table you want to explore
table_name = 'complete_nbbo_20210104'
table_info = conn.get_table(library=library_name, table=table_name)
print(f"Structure of the table '{table_name}' in library '{library_name}':")
print(table_info)

# Don't forget to close the connection when you're done
conn.close()