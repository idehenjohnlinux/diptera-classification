from src.utils.logger import setup_logging
from src.core.dataset_audit import DatasetAudit


def main():

    setup_logging()

    builder = DatasetAudit()

    builder.run()


if __name__ == "__main__":
    main()
