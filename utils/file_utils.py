import time

# nice format for adding date and time to filenames
getdate = lambda : time.asctime().replace(' ', '_').replace(':', '-')
