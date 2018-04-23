# -*- coding: utf-8 -*-

from numpy import *
import sys

def etrs2lonlat(E,N):
    """
    Inputs and outputs: longitude=Easting, latitude=Northing
    see http://www.theseus.fi/bitstream/handle/10024/26222/Manninen_Heli.pdf?sequence=1
    """

    # Meridiaanin pituisen ympyrän säde
    A1 = 6367449.14577105

    # Mittakaavakerroin keskimeridiaanilla
    k0 = 0.9996

    # Itäkoordinaatin arvo keskimeridiaanilla
    E0 = 500000.0

    # Projektion keskimeridiaani (rad)
    km = 0.471238898038469

    # Apusuureita
    s = N / (A1 * k0)
    n = (E - E0) / (A1 * k0)

    # Laskenta
    s1 = 0.000837732168164 * sin(2 * s) * cosh(2 * n)
    s2 = 0.000000059058696 * sin(4 * s) * cosh(4 * n)
    s3 = 0.000000000167349 * sin(6 * s) * cosh(6 * n)
    s4 = 0.000000000000217 * sin(8 * s) * cosh(8 * n)

    n1 = 0.000837732168164 * cos(2 * s) * sinh(2 * n)
    n2 = 0.000000059058696 * cos(4 * s) * sinh(4 * n)
    n3 = 0.000000000167349 * cos(6 * s) * sinh(6 * n)
    n4 = 0.000000000000217 * cos(8 * s) * sinh(8 * n)

    ss = s - (s1 + s2 + s3 + s4)
    nn = n - (n1 + n2 + n3 + n4)

    b = arcsin(sin(ss) / cosh(nn))
    l = arcsin(tanh(nn) / cos(b))

    q = log(tan(b) + sqrt(tan(b)**2 + 1))

    # Ensimmäinen epäkeskeisyys
    e = sqrt(2*1/298.257222101 - 1/298.257222101**2)

    # Jonkinlainen iteraatio
    q1 = q + e * (0.5 * log((1 + e * tanh(q)) / (1 - e * tanh(q))))
    q2 = q + e * (0.5 * log((1 + e * tanh(q1)) / (1 - e * tanh(q1))))
    q3 = q + e * (0.5 * log((1 + e * tanh(q2)) / (1 - e * tanh(q2))))
    q4 = q + e * (0.5 * log((1 + e * tanh(q3)) / (1 - e * tanh(q3))))

    # Lopputuloksena maantieteelliset koordinaatit asteina
    lat = rad2deg(arctan(sinh(q4)))
    lon = rad2deg(km + l)

    return (lon, lat)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        E = float(sys.argv[1])
        N = float(sys.argv[2])
        (lon, lat) = etrs2lonlat(E,N)
        print "(lat, lon) = (%f, %f)" % (lat, lon)
