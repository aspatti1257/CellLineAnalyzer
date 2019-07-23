import threading
import os


class DiagnosticsFileWriter(object):

    FILE_NAME = "Diagnostics.txt"

    @staticmethod
    def writeToFile(input_folder, message, log):
        lock = threading.Lock()
        lock.acquire(True)

        write_action = "w"
        if DiagnosticsFileWriter.FILE_NAME in os.listdir(input_folder):
            write_action = "a"
        with open(input_folder + "/" + DiagnosticsFileWriter.FILE_NAME, write_action) as diagnostics_file:
            try:
                if write_action == "w":
                    diagnostics_file.write("### Diagnostics ###\n")
                diagnostics_file.write(message)
            except ValueError as error:
                log.error("Error writing to file %s. %s", diagnostics_file, error)
            finally:
                diagnostics_file.close()
                lock.release()
