from src.utils import load_projects


def main():
    projects = load_projects()
    print(projects[0]["sections"][3]["content"])


if __name__ == "__main__":
    main()
