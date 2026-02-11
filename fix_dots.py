with open("app.py", "r") as f:
    content = f.read()

OLD = """                        risk = bldg["risk_score"]
                        rc = risk_color(risk)
                        radius = 3 + risk * 8

                        popup_html = (
                            f'<div style="font-family:monospace;font-size:11px;min-width:180px">'
                            f'<b>{bldg.get("address", "N/A")}</b><br>'
                            f'BBL: {bldg.get("bbl", "N/A")}<br>'
                            f'Risk: <span style="color:{rc}">{bldg["risk_label"]} ({risk:.2f})</span><br>'
                            f'Built: {int(bldg["yearbuilt"])} ({int(bldg["building_age"])} yrs)<br>'
                            f'Floors: {int(bldg["numfloors"])} · Units: {int(bldg["unitsres"])}<br>'
                            f'Area: {int(bldg["bldgarea"]):,} sqft<br>'
                            f'Class: {bldg.get("bldgclass", "N/A")}'
                            f'</div>'
                        )

                        folium.CircleMarker(
                            location=[lat, lng],
                            radius=radius,
                            color=rc,
                            fill=True,
                            fill_color=rc,
                            fill_opacity=0.6,
                            popup=folium.Popup(popup_html, max_width=250),
                            tooltip=f'{bldg.get("address", "N/A")} — {bldg["risk_label"]}',
                        ).add_to(bldg_map)"""

NEW = """                        risk = bldg["risk_score"]
                        rc = risk_color(risk)
                        label = bldg["risk_label"]

                        # Visual hierarchy: Critical > High > Moderate > Low
                        if label == "Critical":
                            radius = 10
                            opacity = 0.9
                            border_color = "#FFFFFF"
                            weight = 2
                        elif label == "High":
                            radius = 6
                            opacity = 0.6
                            border_color = rc
                            weight = 1
                        elif label == "Moderate":
                            radius = 4
                            opacity = 0.5
                            border_color = rc
                            weight = 1
                        else:
                            radius = 3
                            opacity = 0.4
                            border_color = rc
                            weight = 1

                        popup_html = (
                            f'<div style="font-family:monospace;font-size:11px;min-width:180px">'
                            f'<b>{bldg.get("address", "N/A")}</b><br>'
                            f'BBL: {bldg.get("bbl", "N/A")}<br>'
                            f'Risk: <span style="color:{rc}">{label} ({risk:.2f})</span><br>'
                            f'Built: {int(bldg["yearbuilt"])} ({int(bldg["building_age"])} yrs)<br>'
                            f'Floors: {int(bldg["numfloors"])} · Units: {int(bldg["unitsres"])}<br>'
                            f'Area: {int(bldg["bldgarea"]):,} sqft<br>'
                            f'Class: {bldg.get("bldgclass", "N/A")}'
                            f'</div>'
                        )

                        folium.CircleMarker(
                            location=[lat, lng],
                            radius=radius,
                            color=border_color,
                            weight=weight,
                            fill=True,
                            fill_color=rc,
                            fill_opacity=opacity,
                            popup=folium.Popup(popup_html, max_width=250),
                            tooltip=f'{bldg.get("address", "N/A")} — {label}',
                        ).add_to(bldg_map)"""

if OLD in content:
    content = content.replace(OLD, NEW)
    
    # Now sort buildings so Critical renders LAST (on top)
    OLD_SORT = """                    for bldg in scored:"""
    NEW_SORT = """                    for bldg in sorted(scored, key=lambda b: b["risk_score"]):"""
    content = content.replace(OLD_SORT, NEW_SORT, 1)
    
    with open("app.py", "w") as f:
        f.write(content)
    
    import subprocess
    r = subprocess.run(["python3", "-c", "import py_compile; py_compile.compile('app.py', doraise=True)"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("[OK] Building dots fixed ✅")
        print("     Critical: large (r=10), white border, high opacity")
        print("     High: medium (r=6), normal opacity")
        print("     Moderate/Low: small, faded")
        print("     Sorted so Critical renders on top")
    else:
        print(f"[ERROR] {r.stderr}")
else:
    print("[ERROR] Target block not found")

