from backend.forget_manager import ForgetManager

fm = ForgetManager()

while True:
    print("\n🔐 Privacy Control Menu")
    print("1. List reflections")
    print("2. List summaries")
    print("3. Forget reflection by keyword")
    print("4. Forget summary by ID (without extension)")
    print("5. Wipe ALL data")
    print("6. Exit")

    choice = input("Select an option: ").strip()

    if choice == "1":
        fm.list_reflections()
    elif choice == "2":
        fm.list_contexts()
    elif choice == "3":
        fm.list_reflections()
        key = input("Enter keyword to forget: ").strip()
        fm.forget_reflection(key)
    elif choice == "4":
        fm.list_contexts()
        sid = input("Enter summary ID (without .json): ").strip()
        fm.forget_context(sid)
    elif choice == "5":
        fm.wipe_all()
    elif choice == "6":
        break
    else:
        print("Invalid option.")
